import dill



with open('model.pkl', 'rb') as f:
    model = dill.load(f)

print(model.parameters())

print(dir())