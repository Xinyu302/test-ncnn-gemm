import re


with open("test_pack_B_handle.cpp", "r") as f:
    content = f.read()

def replace_val_pattern(text):
    # 正则表达式模式，用于匹配类似 _r89ab.val[3] 的字符串
    pattern = r'(_\w+)\.val\[(\d+)\]'
    
    # 使用正则表达式进行替换
    return re.sub(pattern, r'vget_f32m1x4_f32m1(\1, \2)', text)

def replace_vst1q_pattern(text):
    pattern = r'vst1q_f32\(([^,]+), (.+)\);'
    replacement = r'vse32_v_f32m1(\1, \2, VL);'
    new_content = re.sub(pattern, replacement, text)
    return new_content

def replace_vld1q_pattern(text):
    pattern = r'vld1q_f32\(([^)]+)\)'
    replacement = r'vle32_v_f32m1(\1, VL)'
    new_content = re.sub(pattern, replacement, text)
    return new_content

def replace_vld4q_f32_pattern(text):
    pattern = r'vld4q_f32\(([^)]+)\)'
    replacement = r'vlseg4e32_v_f32m1x4(\1, VL)'
    new_content = re.sub(pattern, replacement, text)
    return new_content

def replace_type_float32x4_t(text):
    return text.replace("float32x4_t", "vfloat32m1_t").replace("float32x4x4_t", "vfloat32m1x4_t")

new_content = replace_val_pattern(content)
new_content = replace_type_float32x4_t(new_content)
new_content = replace_vld1q_pattern(new_content)
new_content = replace_vst1q_pattern(new_content)
new_content = replace_vld4q_f32_pattern(new_content)

with open("test_pack_B_handle.cpp", "w") as f:
    f.write(new_content)
