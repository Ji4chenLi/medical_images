HIDDEN_SIZE = 512
dataset = {
            "imgpath" : "/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
            "txtpath" : "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/text_data.pkl",
            "vocabpath" : "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/vocab.pkl",
            "csvpath" : "/data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv",
            "metacsvpath" : "/data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv",
            "mode" : "PER_IMAGE",
            "pathologies" : [
                "Enlarged Cardiomediastinum",
                "Cardiomegaly",
                "Lung Opacity",
                "Lung Lesion",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
                "Pleural Other",
                "Fracture",
                "Support Devices",
                "No Finding"
                ],
            "views": ["PA"],
            "seed": 0,
            "train" : {
                "imgpath" : "/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
                "feature_root": "/home/jiachen.li/data_relu",
                "processed_csv": "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/mimic_train.csv",
                "txtpath": "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/text_data.pkl",
                "vocabpath": "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/vocab.pkl",
                "transforms":[
                    ("Resize", {
                        "size": 256,
                        "interpolation": 1
                    }),
                    ("CenterCrop", {
                        "size": 256
                    }),
                    ("ToTensor", {}),
                    ("Normalize", {
                        "mean": (0.485, 0.456, 0.406),
                        "std": (0.229, 0.224, 0.225)
                    })
                ],
                "mode": "PER_IMAGE",
                "batch_size": 32,
                "lazy_strategy": 'all',
                "cache_strategy": 'none',
                "shuffle": True,
                "shuffle_buffer_size": 32,
                "input_channel": "RGB",
                "num_parallel_calls": 1

            },
            "val" : {
                "imgpath" : "/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
                "feature_root": "/home/jiachen.li/data_relu",
                "processed_csv": "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/mimic_val.csv",
                "txtpath": "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/text_data.pkl",
                "vocabpath": "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/vocab.pkl",
                "transforms":[
                    ("Resize", {
                        "size": 256,
                        "interpolation": 1
                    }),
                    ("CenterCrop", {
                        "size": 256
                    }),
                    ("ToTensor", {}),
                    ("Normalize", {
                        "mean": (0.485, 0.456, 0.406),
                        "std": (0.229, 0.224, 0.225)
                    })
                ],
                "mode": "PER_IMAGE",
                "batch_size": 32,
                "lazy_strategy": 'all',
                "cache_strategy": 'none',
                "shuffle": True,
                "shuffle_buffer_size": 32,
                "input_channel": "RGB",
                "num_parallel_calls": 1,

            },
            "test": {
                "imgpath" : "/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
                "feature_root": "/home/jiachen.li/data_relu",
                "processed_csv": "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/mimic_test.csv",
                "txtpath": "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/text_data.pkl",
                "vocabpath": "/home/jiachen.li/jiachen_medical_images/preprocessed_with_normal/vocab.pkl",
                "transforms": [
                    ("Resize", {
                        "size": 256,
                        "interpolation": 1
                    }),
                    ("CenterCrop", {
                        "size": 256
                    }),
                    ("ToTensor", {}),
                    ("Normalize", {
                        "mean": (0.485, 0.456, 0.406),
                        "std": (0.229, 0.224, 0.225)
                    })
                ],
                "mode": "PER_IMAGE",
                "batch_size": 32,
                "lazy_strategy": 'all',
                "cache_strategy": 'none',
                "shuffle": True,
                "shuffle_buffer_size": 32,
                "input_channel": "RGB",
                "num_parallel_calls": 1,
            },
            "model": {
                "sentence_lstm": {
                    "hidden_size": HIDDEN_SIZE,
                    "num_units": HIDDEN_SIZE,
                    "visual_dim": 1024,
                    "semantic_dim": HIDDEN_SIZE,
                    "num_tags": 14,
                    "top_k_for_semantic": 3,
                },
                "word_lstm":{
                    "hidden_size": HIDDEN_SIZE,
                    "vocab_size": 12432,
                },
                "lambda_stop": 1.,
                "lambda_word": 1.,
                "lambda_attn": 1.,
                "mlc_weights": "exp_default/1611883760.6407511.pt",
            }
        }
