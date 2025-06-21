import json
import os
import re
from pathlib import Path
from datetime import datetime
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

def clean_rules_with_cerebras():        # Use Cerebras (because it's fast and free) to clean and extract core translation rules
    # Load current messy rules database
    with open("rules_database.json", 'r', encoding='utf-8') as f:
        current_db = json.load(f)
    
    print(f"Current database has {len(current_db['rules'])} rules")
    print("Using Cerebras to extract core actionable rules...")
    
    # Initialize Cerebras client
    client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
    
    # Prepare rules text for analysis
    rules_text = ""
    for i, rule in enumerate(current_db["rules"]):
        rules_text += f"\nRule {i+1}:\n"
        rules_text += f"Type: {rule.get('rule_type', 'unknown')}\n"
        rules_text += f"Description: {rule.get('description', '')}\n"
        rules_text += f"Confidence: {rule.get('confidence', 0)}\n"
        if rule.get('examples'):
            rules_text += f"Example: {rule['examples'][0].get('text', '')[:200]}...\n"
        rules_text += "---\n"
    
    prompt = f"""You are a translation expert tasked with cleaning up a messy rules database. Below are translation rules extracted from comparing Chinese cultivation novel translations, but some are parsing artifacts or incomplete.

MESSY RULES DATABASE:
{rules_text}

Extract and rewrite the 8-10 most actionable, high-quality translation rules. Ignore:
- Header text like "Here are the rules..."
- Parsing artifacts with "**" symbols  
- Incomplete sentences
- Vague descriptions
- Example text masquerading as rules

For each rule you keep, provide:
**RULE_TYPE**: [terminology|style|cultural|structure]
**DESCRIPTION**: Clear, actionable principle (one sentence)

Focus on rules that would meaningfully improve future translations. Be selective and prioritize quality.

Output exactly in this format:
1. **terminology**: Use "Alchemy Emperor" instead of "Pill God" for consistency
2. **style**: Prioritize active voice and dynamic verbs in action scenes  
3. **cultural**: Adapt Chinese idioms to natural English while retaining meaning
...etc

Only output the numbered list of rules, nothing else."""

    try:
        # Use qwen-3-32b for reasoning capability and longer context
        response = client.chat.completions.create(
            model="qwen-3-32b",
            messages=[
                {"role": "system", "content": "You are an expert translation analyst specializing in Chinese cultivation novels. Think through what makes a high-quality, actionable translation rule."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Low temperature for consistent extraction
            max_tokens=16382,  # Max tokens for qwen-3-32b for detailed analysis
        )
        
        ai_analysis = response.choices[0].message.content
        print(f"\nCerebras analysis ({len(ai_analysis)} chars):")
        print("=" * 50)
        print(ai_analysis[:500] + "...")
        
        # Save raw Cerebras response
        with open("cerebras_rule_analysis.txt", 'w', encoding='utf-8') as f:
            f.write("Cerebras Rule Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(ai_analysis)
        
        print(f"\nRaw analysis saved to: cerebras_rule_analysis.txt")
        
        # Parse the Cerebras response into structured rules
        clean_rules = parse_cerebras_response(ai_analysis)
        
        # Create final clean database
        final_db = {
            "rules": clean_rules,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_rules": len(clean_rules),
                "cleaned_by": "cerebras_qwen-3-32b",
                "original_count": len(current_db["rules"]),
                "cleaning_date": datetime.now().isoformat()
            }
        }
        
        # Save as both JSON and readable text
        with open("rules_clean.json", 'w', encoding='utf-8') as f:
            json.dump(final_db, f, indent=2, ensure_ascii=False)
        
        # Also save as simple text file for easy reading
        with open("rules_clean.txt", 'w', encoding='utf-8') as f:
            f.write("CLEAN TRANSLATION RULES\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Extracted {len(clean_rules)} high-quality rules from {len(current_db['rules'])} original rules\n")
            f.write(f"Cleaned by: Cerebras qwen-3-32b\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, rule in enumerate(clean_rules, 1):
                f.write(f"RULE {i}: {rule['rule_type'].upper()}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                if 'reasoning' in rule:
                    f.write(f"Reasoning: {rule['reasoning']}\n")
                f.write("\n" + "-"*30 + "\n\n")
        
        print(f"\nFinal results:")
        print(f"Clean rules saved to: rules_clean.json")
        print(f"Readable format saved to: rules_clean.txt")
        print(f"Extracted {len(clean_rules)} rules from {len(current_db['rules'])} original")
        
        # Display the clean rules
        print("=" * 60)
        print("FINAL CLEAN RULES:")
        print("=" * 60)
        
        for i, rule in enumerate(clean_rules, 1):
            print(f"\n{i}. {rule['rule_type'].upper()}: {rule['description']}")
            print(f"   Confidence: {rule['confidence']}")
        
        return final_db
        
    except Exception as e:
        print(f"Error using Cerebras: {e}")
        return None

def parse_cerebras_response(response_text):         # Parse the raw response from Cerebras into structured rules
    rules = []
    
    # Split by numbered lines
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
            
        # Parse format: "1. **terminology**: Use "Alchemy Emperor" instead..."
        if '**' in line and ':' in line:
            try:
                # Extract number
                number_part = line.split('.')[0]
                rest = '.'.join(line.split('.')[1:]).strip()
                
                # Extract rule type (between **)
                if '**' in rest:
                    type_part = rest.split('**')[1].split('**')[0].strip()
                    desc_part = rest.split('**')[2].split(':', 1)[1].strip() if ':' in rest.split('**')[2] else ""
                    
                    if type_part and desc_part:
                        rule = {
                            "id": f"rule_clean_{len(rules)+1}",
                            "rule_type": type_part.lower(),
                            "description": desc_part,
                            "confidence": 0.8,  # Default high confidence since Qwen filtered these
                            "created_at": datetime.now().isoformat(),
                            "usage_count": 0,
                            "success_rate": 0.0,
                            "last_used": None
                        }
                        rules.append(rule)
            except:
                # Skip malformed lines
                continue
    
    return rules

def extract_from_existing_analysis():       # Extract rules from the existing cerebras analysis file
    # Read the existing analysis
    with open("cerebras_rule_analysis.txt", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Extracting rules from existing Cerebras analysis...")
    print(f"Total content length: {len(content)} characters")
    
    # Strip the think tags
    clean_content = re.sub(r'<think>(.*?)</think>', r'\1', content, flags=re.DOTALL)
    clean_content = clean_content.replace("Cerebras Rule Analysis", "").replace("=" * 50, "").strip()
    
    print(f"Content after removing think tags: {len(clean_content)} characters")
    print("\nCleaned content:")
    print("=" * 60)
    print(clean_content)
    
    # Save the cleaned thinking for manual review
    with open("cerebras_thinking_cleaned.txt", 'w', encoding='utf-8') as f:
        f.write("Cleaned Cerebras Thinking Process\n")
        f.write("=" * 40 + "\n\n")
        f.write(clean_content)
    
    print(f"\nCleaned thinking saved to: cerebras_thinking_cleaned.txt")
    print("Review this file to see what Cerebras actually analyzed")
    
    return clean_content

def main():    
    # Check Cerebras API key
    if not os.getenv("CEREBRAS_API_KEY"):
        print("Error: CEREBRAS_API_KEY not found in environment")
        print("Please set your Cerebras API key in .env file")
        return
    
    # Check if rules database exists
    if not Path("rules_database.json").exists():
        print("Error: rules_database.json not found")
        print("Please run the rule extraction pipeline first")
        return
    
    print("Starting Cerebras rule cleaning...")
    print("Using Cerebras qwen-3-32b to extract core rules")
    
    result = clean_rules_with_cerebras()
    
    if result:
        print("\nRule cleaning complete")
        print("Next step: Use rules_clean.json for re-translation testing")
    else:
        print("\nRule cleaning failed")

if __name__ == "__main__":
    main()