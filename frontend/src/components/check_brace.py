with open('src/components/UnifiedReport.tsx', 'r') as f:
    lines = f.readlines()
balance = 0
for i, line in enumerate(lines, 1):
    balance += line.count('{') - line.count('}')
    if balance == 1 and i > 100:
        print(f'First time balance=1 at line {i}: {line.rstrip()[:80]}')
        print(f'  Open braces: {line.count("{")}, Close braces: {line.count("}")}')
        break
print(f'Final balance: {balance}')