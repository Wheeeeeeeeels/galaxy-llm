from typing import Dict, Any

# 模型配置
MODEL_CONFIG: Dict[str, Any] = {
    # 基础模型
    'base_model': {
        'name': 'bert-base-chinese',
        'max_seq_length': 512,
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'initializer_range': 0.02,
        'layer_norm_eps': 1e-12,
        'pad_token_id': 0,
        'bos_token_id': 101,
        'eos_token_id': 102,
        'vocab_size': 21128
    },
    
    # 专家模型
    'expert_models': {
        'math': {
            'name': 'bert-base-chinese',
            'max_seq_length': 512,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'pad_token_id': 0,
            'bos_token_id': 101,
            'eos_token_id': 102,
            'vocab_size': 21128
        },
        'physics': {
            'name': 'bert-base-chinese',
            'max_seq_length': 512,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'pad_token_id': 0,
            'bos_token_id': 101,
            'eos_token_id': 102,
            'vocab_size': 21128
        },
        'chemistry': {
            'name': 'bert-base-chinese',
            'max_seq_length': 512,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'pad_token_id': 0,
            'bos_token_id': 101,
            'eos_token_id': 102,
            'vocab_size': 21128
        },
        'biology': {
            'name': 'bert-base-chinese',
            'max_seq_length': 512,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'pad_token_id': 0,
            'bos_token_id': 101,
            'eos_token_id': 102,
            'vocab_size': 21128
        }
    },
    
    # 路由模型
    'router': {
        'name': 'bert-base-chinese',
        'max_seq_length': 512,
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'initializer_range': 0.02,
        'layer_norm_eps': 1e-12,
        'pad_token_id': 0,
        'bos_token_id': 101,
        'eos_token_id': 102,
        'vocab_size': 21128
    }
}

# 数据配置
DATA_CONFIG: Dict[str, Any] = {
    # 训练数据
    'train': {
        'file_path': 'data/train.json',
        'max_samples': None,
        'batch_size': 4,
        'shuffle': True
    },
    
    # 验证数据
    'eval': {
        'file_path': 'data/eval.json',
        'max_samples': None,
        'batch_size': 4,
        'shuffle': False
    },
    
    # 测试数据
    'test': {
        'file_path': 'data/test.json',
        'max_samples': None,
        'batch_size': 4,
        'shuffle': False
    }
}

# 专家配置
EXPERT_CONFIG = {
    'education': {
        'name': '教育专家',
        'description': '处理教育相关任务，如教学、学习、考试等',
        'capacity': 0.2,
        'special_tasks': [
            '学校信息查询',
            '学校比较',
            '学校推荐',
            '教育政策咨询',
            '学习规划建议'
        ]
    },
    'real_estate': {
        'name': '房产专家',
        'description': '处理房产相关任务，如购房、租房、房产投资等',
        'capacity': 0.2
    },
    'sales': {
        'name': '销售专家',
        'description': '处理销售相关任务，如营销、谈判、客户服务等',
        'capacity': 0.2
    },
    'logic': {
        'name': '逻辑推理专家',
        'description': '处理逻辑推理、问题解决等任务',
        'capacity': 0.2
    },
    'common_sense': {
        'name': '常识问答专家',
        'description': '处理常识性问题和日常知识问答',
        'capacity': 0.2
    }
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 10,
    'eval_steps': 1000,
    'save_steps': 2000,
    'logging_steps': 100,
    'max_seq_length': 512,
    'train_batch_size': 4,
    'eval_batch_size': 4,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 8,
    'fp16': True,
    'fp16_opt_level': 'O1'
}

# 评估配置
EVAL_CONFIG = {
    'metrics': ['accuracy', 'f1', 'rouge'],
    'school_comparison_metrics': ['similarity', 'ranking'],
    'school_recommendation_metrics': ['precision', 'recall', 'ndcg']
} 