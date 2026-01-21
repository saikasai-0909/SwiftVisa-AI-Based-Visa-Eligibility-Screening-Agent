import re
import os

def clean_and_format_text(raw_text):

    # Remove repeated dashes or underscores
    text = re.sub(r'[-_]{3,}', '', raw_text)

    # Split lines and remove empty / numeric-only lines
    lines = text.split('\n')
    filtered_lines = [line.strip() for line in lines if line.strip() and not re.match(r'^\d+$', line.strip())]

    # Merge broken lines
    merged_text = ''
    for line in filtered_lines:
        if re.search(r'[.?!:]$', line):
            merged_text += line + '\n'
        else:
            merged_text += line + ' '

    # Remove extra spaces
    merged_text = re.sub(r'\s+', ' ', merged_text)

    # Restore headings 
    merged_text = re.sub(r'(\d+\.\s[A-Z])', r'\n\n\1', merged_text)

    # Normalize newlines
    merged_text = re.sub(r'\n{2,}', '\n\n', merged_text).strip()

    return merged_text


def process_text_folder(input_folder, output_folder):
    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each text file
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            print(f'Processing {filename}...')
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            clean_text = clean_and_format_text(raw_text)

            out_file = os.path.join(output_folder, filename.replace('.txt', '_clean.txt'))
            # Save cleaned text
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            print(f'Saved cleaned text to {out_file}')

if __name__ == "__main__":
    # Run cleaning pipeline
    process_text_folder('project/cleaned_text', 'project/formatted_texts')