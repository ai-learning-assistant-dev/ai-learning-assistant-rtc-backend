import re


# 辅助函数：检查文本是否超出模型安全处理能力
def is_text_too_complex(text, max_length=100):
    """检查文本是否过于复杂，需要特殊处理"""
    # 检查文本长度
    if len(text) > max_length:
        return True

    # 检查是否包含大量特殊字符或非中文/英文字符
    special_chars = sum(
        1
        for c in text
        if not (c.isalnum() or c.isspace() or c in ".,;:!?，。；：！？、")
    )
    if special_chars > len(text) * 0.2:  # 如果特殊字符超过20%
        return True

    # 检查是否包含大量术语（通常这些术语会很长）
    long_words_count = len([w for w in text.split() if len(w) > 10])
    if long_words_count > 5:
        return True

    return False


# 辅助函数：分段处理文本
def split_text_safely(text, max_length=100):
    """将长文本分成较短的段落，确保模型能够处理"""
    # 先按照常见的句号、问号等分隔符分割
    pattern = r"(?<=[。！？；.!?;])\s*"
    segments = re.split(pattern, text)

    # 过滤空字符串
    segments = [s.strip() for s in segments if s.strip()]

    # 如果分割后的段落仍然太长，进一步分割
    result = []
    for segment in segments:
        if len(segment) <= max_length:
            result.append(segment)
        else:
            # 按逗号、顿号等次要分隔符分割
            sub_pattern = r"(?<=[，、,])\s*"
            sub_segments = re.split(sub_pattern, segment)
            sub_segments = [s.strip() for s in sub_segments if s.strip()]

            # 如果二次分割的段落仍然太长，按固定长度分割
            for sub in sub_segments:
                if len(sub) <= max_length:
                    result.append(sub)
                else:
                    # 按固定长度分割，但尽量不打断词语
                    for i in range(0, len(sub), max_length - 10):
                        if i + max_length < len(sub):
                            result.append(sub[i : i + max_length - 10])
                        else:
                            result.append(sub[i:])

    # 合并短段落，使得每段尽可能接近max_length
    merged_result = []
    current_segment = ""

    for segment in result:
        # 如果当前段落加上新段落不超过最大长度，则合并
        if len(current_segment) + len(segment) + 1 <= max_length:
            if current_segment:
                current_segment += " " + segment
            else:
                current_segment = segment
        else:
            # 如果当前段落不为空，添加到结果中
            if current_segment:
                merged_result.append(current_segment)
            # 如果当前段落本身就超过最大长度，直接添加
            if len(segment) >= max_length:
                merged_result.append(segment)
                current_segment = ""
            else:
                current_segment = segment

    # 添加最后一个段落
    if current_segment:
        merged_result.append(current_segment)

    return merged_result
