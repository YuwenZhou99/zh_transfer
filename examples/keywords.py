import argparse
import inseq
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM

def read_lines_from_file(file_path):
    """读取文件并返回每一行内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_output_file(file_path, content):
    """将内容写入输出文件"""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def main(input_file, generated_text1_file, generated_text2_file):
    causal_lm_names = "01-ai/Yi-6B"
    
    model_name = AutoModelForCausalLM.from_pretrained(causal_lm_names)
    tokenizer = AutoTokenizer.from_pretrained(causal_lm_names)
    
    model = inseq.load_model(model_name, "integrated_gradients", tokenizer=tokenizer)
    
    input_lines = read_lines_from_file(input_file)
    generated_text1_lines = read_lines_from_file(generated_text1_file)
    generated_text2_lines = read_lines_from_file(generated_text2_file)
    
    for i, input_text in enumerate(input_lines):
        try:
            input_text = input_text.strip()
            generated_text1 = generated_text1_lines[i].strip()
            generated_text2 = generated_text2_lines[i].strip()
            
            # 调用模型进行属性计算
            out = model.attribute(
                input_texts=[input_text, input_text],
                # generated_texts 用于强制属性计算指定的文本
                generated_texts=[generated_text1, generated_text2],
                step_scores=["probability"],
                skip_special_tokens=True,
            )
            
            out.show()
        except Exception as e:
            print(f"Skipping line {i} due to error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some texts.')
    parser.add_argument('input_file', type=str, help='The input text file for generation')
    parser.add_argument('generated_text1_file', type=str, help='The first generated text file for attribution')
    parser.add_argument('generated_text2_file', type=str, help='The second generated text file for attribution')

    args = parser.parse_args()
    main(args.input_file, args.generated_text1_file, args.generated_text2_file)
