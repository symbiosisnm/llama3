import argparse
import json
import re
import html
from typing import Dict


def clean_text(text: str) -> str:
    """
    Clean input text by removing HTML tags, emails, phone numbers, and extra whitespace.
    Replaces sensitive tokens with placeholders to help anonymize data.
    """
    # Unescape HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', text)
    # Replace phone numbers (simple US format)
    text = re.sub(r'\b\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}\b', '[PHONE]', text)
    # Collapse multiple whitespace characters
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def process_jsonl(input_path: str, output_path: str) -> None:
    """
    Read a JSONL file containing records with 'prompt' and 'response' fields, clean the text,
    and write the cleaned records to a new JSONL file.
    """
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            record: Dict = json.loads(line)
            for field in ['prompt', 'response']:
                if field in record and isinstance(record[field], str):
                    record[field] = clean_text(record[field])
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Clean and preprocess business data for model training.')
    parser.add_argument('--input', required=True, help='Path to the input JSONL file containing raw data.')
    parser.add_argument('--output', required=True, help='Path to write the cleaned JSONL file.')
    args = parser.parse_args()
    process_jsonl(args.input, args.output)


if __name__ == '__main__':
    main()
