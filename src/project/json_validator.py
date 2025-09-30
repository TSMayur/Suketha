# src/project/json_validator.py

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_and_inspect_json(file_path: str):
    """
    Validate JSON file and show detailed error information
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📁 Checking file: {path}")
    print(f"📊 File size: {path.stat().st_size / (1024*1024):.2f} MB")
    
    # First, let's read the raw content to see what's actually there
    print("\n--- RAW CONTENT INSPECTION ---")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"Total characters: {len(content)}")
        print("First 200 characters:")
        print(repr(content[:200]))
        print("\nFirst 200 characters (pretty):")
        print(content[:200])
        
        # Check for common JSON issues
        print("\n--- STRUCTURAL CHECKS ---")
        if content.startswith('{\n  "rows": [\n'):
            print("✅ Starts correctly with JSON structure")
        else:
            print(f"❌ Unexpected start. Expected: {repr('{\n  \"rows\": [\n')}")
            print(f"❌ Actually starts with: {repr(content[:20])}")
        
        if content.endswith('\n  ]\n}\n'):
            print("✅ Ends correctly with JSON structure")
        else:
            print("❌ Does not end with proper JSON structure")
            print(f"Last 50 characters: {repr(content[-50:])}")
        
        # Look for potential issues
        print("\n--- POTENTIAL ISSUES ---")
        
        # Check for trailing commas
        lines = content.split('\n')
        for i, line in enumerate(lines[-10:], start=len(lines)-10):
            if line.strip().endswith(',') and i >= len(lines) - 3:
                print(f"⚠️  Potential trailing comma at line {i+1}: {repr(line.strip())}")
        
        # Check for missing commas
        brace_lines = [i for i, line in enumerate(lines) if line.strip().startswith('}')]
        for i in brace_lines[:5]:  # Check first few closing braces
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('{') and not lines[i].strip().endswith(','):
                    print(f"⚠️  Missing comma after closing brace at line {i+1}")
        
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return
    
    # Now try to parse as JSON
    print("\n--- JSON PARSING TEST ---")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("✅ JSON is valid!")
        print(f"Root keys: {list(data.keys())}")
        
        if 'rows' in data:
            rows = data['rows']
            print(f"Number of rows: {len(rows)}")
            
            if len(rows) > 0:
                first_record = rows[0]
                print(f"First record keys: {list(first_record.keys())}")
                
                # Check for the vector field specifically
                if 'embedding' in first_record:
                    embedding = first_record['embedding']
                    if isinstance(embedding, list):
                        print(f"✅ Embedding field: list with {len(embedding)} dimensions")
                    else:
                        print(f"❌ Embedding field: {type(embedding)} (should be list)")
                else:
                    print("❌ No 'embedding' field found in first record")
            else:
                print("❌ No rows in data")
        else:
            print("❌ No 'rows' key in JSON")
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed!")
        print(f"Error: {e}")
        print(f"Line {e.lineno}, Column {e.colno}")
        print(f"Position in file: {e.pos}")
        
        # Try to show the problematic area
        try:
            lines = content.split('\n')
            error_line = e.lineno - 1
            start_line = max(0, error_line - 2)
            end_line = min(len(lines), error_line + 3)
            
            print(f"\nProblematic area (lines {start_line + 1}-{end_line}):")
            for i in range(start_line, end_line):
                marker = " >>> " if i == error_line else "     "
                print(f"{marker}Line {i+1:3d}: {repr(lines[i])}")
        except:
            print("Could not show problematic area")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def fix_common_json_issues(input_file: str, output_file: str = None):
    """
    Attempt to fix common JSON issues
    """
    if output_file is None:
        output_file = input_file + ".fixed"
    
    print(f"\n--- ATTEMPTING TO FIX JSON ---")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove trailing commas before closing brackets/braces
        import re
        
        # Fix trailing comma before closing array bracket
        content = re.sub(r',(\s*\])', r'\1', content)
        
        # Fix trailing comma before closing object brace  
        content = re.sub(r',(\s*\})', r'\1', content)
        
        # Write fixed version
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed version written to: {output_file}")
        
        # Test the fixed version
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print("✅ Fixed JSON is valid!")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ Fixed JSON still invalid: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to fix JSON: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    file_path = sys.argv[1] if len(sys.argv) > 1 else "prepared_for_upload/prepared_data.json"
    
    print("=" * 60)
    print("JSON VALIDATOR AND INSPECTOR")
    print("=" * 60)
    
    validate_and_inspect_json(file_path)
    
    # Ask if user wants to try fixing
    try:
        response = input("\nDo you want to attempt automatic fixing? (y/n): ").lower().strip()
        if response == 'y':
            if fix_common_json_issues(file_path):
                print("\n--- TESTING FIXED FILE ---")
                validate_and_inspect_json(file_path + ".fixed")
    except KeyboardInterrupt:
        print("\nAborted.")