with open('UnifiedReport.tsx', 'r') as f:
    lines = f.readlines()
balance = 0
for i, line in enumerate(lines, 1):
    balance += line.count('{') - line.count('}')
    if balance == 1 and i > 189:
        print(f'Line {i}: balance goes to 1: {line.rstrip()[:80]}')
        print(f'  open={line.count("{")}, close={line.count("}")}')
        break
print(f'Final balance: {balance}')