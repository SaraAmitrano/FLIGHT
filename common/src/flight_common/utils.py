def get_full_class_name(obj: object) -> str:
    module = obj.__class__.__module__
    if module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + "." + obj.__class__.__name__
