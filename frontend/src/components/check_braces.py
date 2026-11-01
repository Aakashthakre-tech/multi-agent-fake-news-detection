with open('src/components/UnifiedReport.tsx', 'r') as f:
    lines = f.readlines()
balance = 0
for i, line in enumerate(lines, 1):
    balance += line.count('{') - line.count('}')
    if balance < 0:
        print(f'Line {i}: NEGATIVE balance: {line.rstrip()}')
        break
print(f'Final balance: {balance}')
if balance > 0:
    print('Missing closing brace(s) somewhere')
    balance = 0
    first_positive = None
    for i, line in enumerate(lines, 1):
        old_balance = balance
        balance += line.count('{') - line.count('}')
        if balance > 0 and old_balance <= 0 and first_positive is None:
            first_positive = i
        print(f'Line {i}: bal={balance} ({line.count("{")} open, {line.count("}")} close) {line.rstrip()[:80]}')