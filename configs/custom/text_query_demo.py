"""
커스텀 3DGS 장면에서 텍스트 쿼리로 하이라이트하는 데모용 설정

사용법:
    python tools/test.py \
        --config-file configs/custom/text_query_demo.py \
        --options \
            test_only=True \
            test.skip_eval=True \
            test.save_feat=True \
            data.test.data_root=/path/to/scenes \
            data.test.split=your_scene_folder \
            weight=/path/to/checkpoint.pth \
            save_path=/path/to/output
"""

_base_ = [
    "../../_base_/default_runtime.py",
]

# 기본 설정
repo_root = "/home/stevenlair/LAIR/AGI-Memory/SceneSplat"
gpu_nums = 1
batch_size_test = 1
num_worker = 4
test_only = True

# 모델 설정 (기본 lang-pretrain 모델 구조)
model = dict(
    type="LangPretrainer",
    backbone=dict(
        type="PT-v3m1",
        in_channels=11,  # color 3, quat 4, scale 3, opacity 1
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 6),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2),
        dec_channels=(768, 512, 256),
        dec_num_head=(16, 16, 16),
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CosineSimilarity", reduction="mean", loss_weight=1.0),
        dict(type="L2Loss", reduction="mean", loss_weight=1.0),
        dict(
            type="AggregatedContrastiveLoss",
            temperature=0.2,
            reduction="mean",
            loss_weight=0.025,
            schedule="all",
        ),
    ],
)

# 데이터셋 설정
data = dict(
    num_classes=-1,  # 평가 안 하므로 무관
    ignore_index=-1,
    test=dict(
        type="GenericGSDataset",
        split="demo_scene",  # data_root 아래의 폴더 이름
        data_root="/media/stevenlair/Data_exFAT/SceneSplat/custom_scenes",  # 여기 수정 필요
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(
                type="Copy",
                keys_dict=dict(
                    coord="origin_coord",
                ),
            ),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "opacity", "quat", "scale"),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "opacity", "quat", "scale"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "opacity", "quat", "scale"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ]
            ],
        ),
    ),
)

# Tester 설정 (특징만 저장, 평가는 스킵)
test = dict(
    type="ZeroShotSemSegTester",
    verbose=True,
    class_names=None,  # 평가 안 하므로 None
    text_embeddings=None,  # 평가 안 하므로 None
    excluded_classes=None,
    enable_voting=False,  # 특징만 저장하므로 불필요
    vote_k=25,
    confidence_threshold=0.1,
    save_feat=True,  # 중요: 특징 저장 활성화
    skip_eval=True,  # 중요: 평가 스킵
)

# 체크포인트 경로 (--options로 덮어쓸 수 있음)
weight = "/media/stevenlair/Data_exFAT/SceneSplat/checkpoints/lang-pretrain-concat/model_best.pth"

# 저장 경로 (--options로 덮어쓸 수 있음)
save_path = "/media/stevenlair/Data_exFAT/SceneSplat/runs/text_query_demo"

