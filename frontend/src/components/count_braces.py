with open('UnifiedReport.tsx', 'r') as f:
    content = f.read()
opens = content.count('{')
closes = content.count('}')
print(f'Open braces: {opens}')
print(f'Close braces: {closes}')
print(f'Difference: {opens - closes}')
if opens > closes:
    balance = 0
    for i, line in enumerate(content.splitlines(), 1):
        balance += line.count('{') - line.count('}')
        if balance == 1:
            print(f'First time balance=1 at line {i}: {line.rstrip()[:80]}')
            break
    print(f'Final balance: {balance}')