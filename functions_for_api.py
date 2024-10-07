import pickle


# load model and encoder
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('ordinal_encoder.pkl', 'rb') as file:
    oe = pickle.load(file)


def input_transformation(data):
    obj_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    to_transform = [val[0] for key,val in data.items() if key in obj_columns]
    to_transform = oe.transform([to_transform])[0]
    transformed = {col:val for col,val in zip(obj_columns, to_transform)}

    data = [float(data[key][0]) if key not in obj_columns else transformed[key] for key in data.keys()]

    return data

def output_transformation(output):
    classes = {0:"Don't have heart disease", 1:"Have heart disease"}

    return classes[output]


def classify(data):
    data = input_transformation(data)
    
    result = model.predict([data])[0]
    
    return output_transformation(result)


def classify_fastapi(data):
    for key in data.keys():
        data[key] = [data[key]]
    data = input_transformation(data)

    prediction = model.predict([data])[0]
    
    return output_transformation(prediction)