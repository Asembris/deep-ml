def mmlu_letter_matching(model_outputs: list[str], ground_truth: list[str], subjects: list[str]) -> dict:
    """
    Evaluate MMLU predictions using letter-matching.
    
    Args:
        model_outputs: List of model generated responses
        ground_truth: List of correct answer letters (A, B, C, or D)
        subjects: List of subject names for each question
    
    Returns:
        Dictionary with evaluation metrics
    """
    import re
    def extract_answer(s):
        match = re.search(r'\b([A-D])\b', s, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None


    total_questions=len(model_outputs)
    unique_subjects=list(set(subjects))
    cnt_subjects={e:0 for e in unique_subjects}
    for e in subjects:
        cnt_subjects[e]+=1
    
    subject_accuracy={e:0 for e in unique_subjects}
    total_correct=0
    valid_response_rate=0
    for (i,output) in enumerate(model_outputs):
        res=extract_answer(output)
        if res is not None and res==ground_truth[i]:
            total_correct+=1
            subject_accuracy[subjects[i]]+=1
            valid_response_rate+=1
        elif res is not None:
            valid_response_rate+=1
    for e in subject_accuracy:
        subject_accuracy[e]/=cnt_subjects[e]
    overall_accuracy=total_correct/total_questions
    valid_response_rate/=total_questions
    res={"overall_accuracy":overall_accuracy,"subject_accuracy":subject_accuracy
    ,"valid_response_rate":valid_response_rate,"total_correct":total_correct,"total_questions":total_questions
    }
    return res










