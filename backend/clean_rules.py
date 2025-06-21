import json
import os
from pathlib import Path
from datetime import datetime
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

def clean_rules_with_cerebras():
    """Use Cerebras to intelligently clean and extract core translation rules"""
    
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

Extract and rewrite ONLY the 5 most actionable, high-quality translation rules. Ignore:
- Header text like "Here are the rules..."
- Parsing artifacts with "**" symbols
- Incomplete sentences
- Vague descriptions
- Example text masquerading as rules

For each rule you keep, provide:
1. **RULE_TYPE**: [terminology|style|cultural|structure]
2. **DESCRIPTION**: Clear, actionable principle (one sentence)
3. **CONFIDENCE**: [high|medium|low]
4. **REASONING**: Why this rule is important for cultivation novel translation

Focus on rules that would meaningfully improve future translations. Be selective and prioritize quality over quantity.

Output format:
RULE_1:
TYPE: [type]
DESCRIPTION: [clear actionable rule]
CONFIDENCE: [level]
REASONING: [why this matters]

RULE_2:
...etc
"""

    try:
        # Use llama-3.3-70b for good reasoning capability
        response = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {"role": "system", "content": "You are an expert translation analyst specializing in Chinese cultivation novels. Extract only the highest quality, most actionable translation rules."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=2000
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
                "cleaned_by": "cerebras_llama-3.3-70b",
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
            f.write(f"Cleaned by: Cerebras llama-3.3-70b\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, rule in enumerate(clean_rules, 1):
                f.write(f"RULE {i}: {rule['rule_type'].upper()}\n")
                f.write(f"Description: {rule['description']}\n")
                f.write(f"Confidence: {rule['confidence']}\n")
                if 'reasoning' in rule:
                    f.write(f"Reasoning: {rule['reasoning']}\n")
                f.write("\n" + "-"*30 + "\n\n")
        
        print(f"\nFinal results:")
        print(f"✓ Clean rules saved to: rules_clean.json")
        print(f"✓ Readable format saved to: rules_clean.txt")
        print(f"✓ Extracted {len(clean_rules)} rules from {len(current_db['rules'])} original")
        
        # Display the clean rules
        print(f"\n{'='*60}")
        print("FINAL CLEAN RULES:")
        print(f"{'='*60}")
        
        for i, rule in enumerate(clean_rules, 1):
            print(f"\n{i}. {rule['rule_type'].upper()}: {rule['description']}")
            print(f"   Confidence: {rule['confidence']}")
        
        return final_db
        
    except Exception as e:
        print(f"Error using Cerebras: {e}")
        return None

def parse_cerebras_response(response_text):
    """Parse Cerebras response into structured rules"""
    rules = []
    
    # Split by RULE_ markers
    sections = response_text.split("RULE_")[1:]  # Skip first empty section
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        rule = {
            "id": f"rule_clean_{i+1}",
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "success_rate": 0.0,
            "last_used": None
        }
        
        lines = section.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("TYPE:"):
                rule_type = line.replace("TYPE:", "").strip().replace("[", "").replace("]", "")
                rule["rule_type"] = rule_type
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
                rule["description"] = description
            elif line.startswith("CONFIDENCE:"):
                conf_text = line.replace("CONFIDENCE:", "").strip().replace("[", "").replace("]", "").lower()
                confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                rule["confidence"] = confidence_map.get(conf_text, 0.7)
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
                rule["reasoning"] = reasoning
        
        # Validate rule has essential fields
        if "rule_type" in rule and "description" in rule:
            rules.append(rule)
    
    return rules

def main():
    """Main entry point"""
    
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
    print("Using free Cerebras llama-3.3-70b to extract core rules")
    
    result = clean_rules_with_cerebras()
    
    if result:
        print("\n✅ Rule cleaning complete!")
        print("Next step: Use rules_clean.json for re-translation testing")
    else:
        print("\n❌ Rule cleaning failed")

if __name__ == "__main__":
    main()