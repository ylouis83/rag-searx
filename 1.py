import re

def is_definitely_duplicate(address, refer):
    # 规则1: 完全相等
    if address == refer:
        return True
    # 规则2: refer以address开头
    if refer.startswith(address):
        return True
    # 规则3: 拼接后完美重复
    full_address = address + refer
    if len(full_address) > 2 and len(full_address) % 2 == 0:
        if full_address[:len(full_address)//2] == full_address[len(full_address)//2:]:
            return True
    return False
