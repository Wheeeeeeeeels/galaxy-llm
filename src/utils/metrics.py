import numpy as np
from typing import List, Dict, Any, Union
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from rouge import Rouge

def compute_metrics(
    predictions: List[int],
    labels: List[int],
    metrics: List[str] = ['accuracy', 'f1', 'precision', 'recall']
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        metrics: 要计算的指标列表
        
    Returns:
        指标字典
    """
    # 转换为numpy数组
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # 计算指标
    results = {}
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(labels, predictions)
        
    if 'f1' in metrics:
        results['f1'] = f1_score(labels, predictions, average='weighted')
        
    if 'precision' in metrics:
        results['precision'] = precision_score(labels, predictions, average='weighted')
        
    if 'recall' in metrics:
        results['recall'] = recall_score(labels, predictions, average='weighted')
        
    return results

def compute_school_comparison_metrics(
    predictions: List[Dict[str, Any]],
    labels: List[Dict[str, Any]],
    metrics: List[str]
) -> Dict[str, float]:
    """
    计算学校比较指标
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        metrics: 要计算的指标列表
        
    Returns:
        指标字典
    """
    results = {}
    
    for metric in metrics:
        if metric == 'similarity':
            # 计算预测和标签的相似度
            similarities = []
            for pred, label in zip(predictions, labels):
                pred_text = ' '.join(str(v) for v in pred.values())
                label_text = ' '.join(str(v) for v in label.values())
                similarity = compute_text_similarity(pred_text, label_text)
                similarities.append(similarity)
            results['similarity'] = np.mean(similarities)
            
        elif metric == 'ranking':
            # 计算排名准确率
            correct_rankings = 0
            total_rankings = 0
            
            for pred, label in zip(predictions, labels):
                pred_ranking = sorted(
                    pred.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                label_ranking = sorted(
                    label.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                if pred_ranking == label_ranking:
                    correct_rankings += 1
                total_rankings += 1
                
            results['ranking_accuracy'] = correct_rankings / total_rankings
            
    return results

def compute_school_recommendation_metrics(
    predictions: List[List[str]],
    labels: List[List[str]],
    metrics: List[str],
    k: int = 5
) -> Dict[str, float]:
    """
    计算学校推荐指标
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        metrics: 要计算的指标列表
        k: 推荐数量
        
    Returns:
        指标字典
    """
    results = {}
    
    for metric in metrics:
        if metric == 'precision':
            # 计算精确率
            precisions = []
            for pred, label in zip(predictions, labels):
                pred_set = set(pred[:k])
                label_set = set(label)
                if len(pred_set) > 0:
                    precision = len(pred_set & label_set) / len(pred_set)
                    precisions.append(precision)
            results['precision@k'] = np.mean(precisions)
            
        elif metric == 'recall':
            # 计算召回率
            recalls = []
            for pred, label in zip(predictions, labels):
                pred_set = set(pred[:k])
                label_set = set(label)
                if len(label_set) > 0:
                    recall = len(pred_set & label_set) / len(label_set)
                    recalls.append(recall)
            results['recall@k'] = np.mean(recalls)
            
        elif metric == 'ndcg':
            # 计算NDCG
            ndcgs = []
            for pred, label in zip(predictions, labels):
                dcg = 0
                for i, school in enumerate(pred[:k]):
                    if school in label:
                        dcg += 1 / np.log2(i + 2)
                        
                idcg = 0
                for i in range(min(k, len(label))):
                    idcg += 1 / np.log2(i + 2)
                    
                if idcg > 0:
                    ndcg = dcg / idcg
                    ndcgs.append(ndcg)
                    
            results['ndcg@k'] = np.mean(ndcgs)
            
    return results

def compute_text_similarity(text1: str, text2: str) -> float:
    """
    计算文本相似度
    
    Args:
        text1: 文本1
        text2: 文本2
        
    Returns:
        相似度
    """
    # 使用简单的词重叠率作为相似度
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
        
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union 