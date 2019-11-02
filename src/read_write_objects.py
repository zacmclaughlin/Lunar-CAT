import pickle
import os


def fileparts(file):
    path, filename = os.path.split(file)
    name, extension = filename.split('.')
    return path, name, extension


def save_obj_to_file(filename, obj):
    path, name, extension = fileparts(filename)

    if not os.path.isdir(path):
        os.mkdir(path)

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_object_from_file(filename):

    obj = pickle.load(open(filename, 'rb'))

    return obj

#
# def load_output_from_file(path_to_model, model_name):
#
#     output_filename = path_to_model + model_name
#     output = read_object_from_file(output_filename)
#     return output
