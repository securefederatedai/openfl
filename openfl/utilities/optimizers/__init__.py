if pkgutil.find_loader('torch'):
    import .torch
if pkgutil.find_loader('keras'):
    import .keras
