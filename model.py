import cv2
import rff
import clip
import torch
import antialiased_cnns
import numpy as np
import torch.nn as nn

from utils import generate_random_rotations, unnormalize_image, visualize_so3_probabilities, generate_equivolumetric_grid

# torch.Size([3, 224, 224])
# torch.Size([3, 224, 224])
# torch.Size([3, 3])




class RelPose(nn.Module):
    def __init__(
        self,
        use_clip=True,
        feature_extractor=None,
        num_pe_bases=8,
        num_layers=4,
        hidden_size=256,
        recursion_level=3,
        num_queries=50000,
        sample_mode="equivolumetric",
        num_images=2,
        metadata_size=0,
        freeze_encoder=False,
    ):
        """
        Args:
            feature_extractor (nn.Module): Feature extractor.
            num_pe_bases (int): Number of positional encoding bases.
            num_layers (int): Number of layers in the network.
            hidden_size (int): Size of the hidden layer.
            recursion_level (int): Recursion level for healpix if using equivolumetric
                sampling.
            num_queries (int): Number of rotations to sample if using random sampling.
            sample_mode (str): Sampling mode. Can be equivolumetric or random.
        """
        super().__init__()
        
        self.use_positional_encoding = num_pe_bases > 0
        if self.use_positional_encoding:
            query_size = num_pe_bases * 2 * 9
            metadata_size = num_pe_bases * 2 * metadata_size
            self.register_buffer(
                "embedding", (2 ** torch.arange(num_pe_bases)).reshape(1, 1, -1)
            )
        else:
            query_size = 9
            
        
        self.use_clip = use_clip
        if not use_clip:
            model = antialiased_cnns.resnet50(pretrained=True) # 2048
            feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
            self.feature_extractor_params = model.parameters()
            self.feature_extractor_statedict = model.state_dict()
            self.embed_feature = nn.Linear(2048 * num_images, hidden_size)
            self.embed_query = nn.Linear(query_size, hidden_size)
        else:
            model, _ = clip.load("ViT-B/32", device=torch.device("cpu")) # 512
            model.cuda()
            self.feature_extractor_params = model.parameters()
            self.feature_extractor_statedict = model.state_dict()
            feature_extractor = model.encode_image
            self.embed_feature = nn.Linear(512 * num_images, hidden_size)
            self.embed_query = nn.Linear(query_size, hidden_size)
            
        self.recursion_level = recursion_level
        self.num_queries = num_queries
        self.sample_mode = sample_mode
        self.num_images = num_images
        self.metadata_size = metadata_size

        self.feature_extractor = feature_extractor
        if freeze_encoder:
            self.freeze_encoder()


        if self.metadata_size > 0:
            self.embed_metadata = nn.Linear(metadata_size, hidden_size)
        layers = []
        for _ in range(num_layers - 2):
            layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)
        self.equi_grid = {}

    def freeze_encoder(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def positional_encoding(self, x):
        """
        Args:
            x (tensor): Input (B, D).

        Returns:
            y (tensor): Positional encoding (B, 2 * D * L).
        """
        if not self.use_positional_encoding:
            return x
        embed = (x[..., None] * self.embedding).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

    def forward(
        self,
        images1=None,
        images2=None,
        features1=None,
        features2=None,
        gt_rotation=None,
        recursion_level=None,
        num_queries=None,
        queries=None,
        images3=None,
        metadata=None,
    ):
        """
        Args:
            images1 (tensor): First set of images (B, 3, 224, 224).
            images2 (tensor): Corresponding set of images (B, 3, 224, 224).
            gt_rotation (tensor): Ground truth rotation (B, 3, 3).
            num_queries (int): Number of rotations to sample if using random sampling.

        Returns:
            rotations (tensor): Rotation matrices (B, num_queries, 3, 3). First query
                is the ground truth rotation.
            logits (tensor): logits (B, num_queries).
        """

        if features1 is None:
            features1 = self.feature_extractor(images1) # (B, 2048, 1, 1)
        if features2 is None:
            features2 = self.feature_extractor(images2)# (B, 2048, 1, 1)
        if images3 is not None:
            features3 = self.feature_extractor(images3)
            features = torch.cat([features1, features2, features3], dim=1) # (B, 3*2048, 1, 1)
        else:
            features = torch.cat([features1, features2], dim=1) # (B, 2*2048, 1, 1)

        batch_size = features1.size(0) # B
        assert batch_size == features2.size(0)
        features = features.reshape(batch_size, -1)  # (B, 4096)
        if queries is None:
            if self.sample_mode == "equivolumetric":
                if recursion_level is None:
                    recursion_level = self.recursion_level
                if recursion_level not in self.equi_grid:
                    self.equi_grid[recursion_level] = generate_equivolumetric_grid(
                        recursion_level
                    )
                queries = self.equi_grid[recursion_level].to(images1.device)
                num_queries = len(queries)
            elif self.sample_mode == "random":
                if num_queries is None:
                    num_queries = self.num_queries
                queries = generate_random_rotations(num_queries, device=images1.device)
            else:
                raise Exception(f"Unknown sampling mode {self.sample_mode}.")

            if gt_rotation is not None:
                delta_rot = queries[0].T @ gt_rotation
                # First entry will always be the gt rotation
                queries = torch.einsum("aij,bjk->baik", queries, delta_rot)
            else:
                if len(queries.shape) == 3:
                    queries = queries.unsqueeze(0)
                num_queries = queries.shape[1]
        else:
            num_queries = queries.shape[1]

        queries_pe = self.positional_encoding(queries.reshape(-1, num_queries, 9))

        if metadata is not None:
            metadata = self.positional_encoding(
                metadata.reshape(-1, self.metadata_size)
            )
            e_m = self.embed_metadata(metadata).unsqueeze(1)  # (B, 1, H)
        else:
            e_m = 0
            if self.metadata_size > 0:
                raise Warning("Metadata size is non-zero but no metadata is provided.")
        e_f = self.embed_feature(features).unsqueeze(1)  # (B, 1, H)
        e_q = self.embed_query(queries_pe)  # (B, n_q, H)
        out = self.layers(e_f + e_q + e_m)  # (B, n_q, 1)
        logits = out.reshape(batch_size, num_queries)
        # logits = logits.softmax(dim=1)
        return queries, logits

    def predict_probability(
        self, images1, images2, query_rotation, recursion_level=None, num_queries=None
    ):
        """
        Args:
            images1 (tensor): First set of images (B, 3, 224, 224).
            images2 (tensor): Corresponding set of images (B, 3, 224, 224).
            gt_rotation (tensor): Ground truth rotation (B, 3, 3).
            num_queries (int): Number of rotations to sample. If gt_rotation is given
                will sample num_queries - batch size.

        Returns:
            probabilities
        """
        logits = self.forward(
            images1,
            images2,
            gt_rotation=query_rotation,
            num_queries=num_queries,
            recursion_level=recursion_level,
        )
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = probabilities * num_queries / np.pi ** 2
        return probabilities[:, 0]

    def save_model(self, path):
        save_dict = {
            "embed_feature_state_dict": self.embed_feature.state_dict(),
            "embed_query_state_dict": self.embed_query.state_dict(),
            "feature_extractor_state_dict": self.feature_extractor_statedict,
            "clip": self.use_clip,
        }
        torch.save(save_dict, path)


    def load_model(self, path, load_metadata=True):
        print(path)            
        if not self.use_clip:
            self.feature_extractor.load_state_dict(save_dict["feature_extractor_state_dict"], strict=True)
        else:
            save_dict = torch.load(path, map_location='cuda:0')
            _model, _ = clip.load("ViT-B/32", device=torch.device("cpu")) # 512
            _model.cuda()
            _model_dict = _model.state_dict()
            _model_dict.update(save_dict['feature_extractor_state_dict'])
            _model.load_state_dict(_model_dict)
            self.feature_extractor = _model.encode_image
            
        self.embed_feature.load_state_dict(save_dict["embed_feature_state_dict"], strict=True)
        self.embed_query.load_state_dict(save_dict["embed_query_state_dict"], strict=True)
        # self.embed_feature.eval()
        # self.embed_query.eval()
        # self.layers.eval()

    def make_visualization(
        self,
        images1,
        images2,
        rotations,
        probabilities,
        num_vis=5,
        model_id=None,
        category=None,
        ind1=None,
        ind2=None,
    ):
        images1 = images1[:num_vis].detach().cpu().numpy().transpose(0, 2, 3, 1)
        images2 = images2[:num_vis].detach().cpu().numpy().transpose(0, 2, 3, 1)
        rotations = rotations[:num_vis].detach().cpu().numpy()
        probabilities = probabilities[:num_vis].detach().cpu().numpy()
        visuals = []
        for i in range(len(images1)):
            image1 = unnormalize_image(cv2.resize(images1[i], (448, 448)))
            image2 = unnormalize_image(cv2.resize(images2[i], (448, 448)))
            so3_vis = visualize_so3_probabilities(
                rotations=rotations[i],
                probabilities=probabilities[i],
                rotations_gt=rotations[i, 0],
                to_image=True,
                display_threshold_probability=1 / len(probabilities[i]),
            )
            full_image = np.vstack((np.hstack((image1, image2)), so3_vis))
            if model_id is not None:
                cv2.putText(full_image, model_id[i], (5, 40), 4, 1, (0, 0, 255))
                cv2.putText(full_image, category[i], (5, 80), 4, 1, (0, 0, 255))
                cv2.putText(full_image, str(int(ind1[i])), (5, 120), 4, 1, (0, 0, 255))
                cv2.putText(
                    full_image, str(int(ind2[i])), (453, 120), 4, 1, (0, 0, 255)
                )
            visuals.append(full_image)
        
        return visuals





