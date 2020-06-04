import os


def absolute_path_from_home(relative_path=None):
    home_path = os.path.dirname(__file__).split("/")[:-1]

    if relative_path == None:
        return "/".join(home_path)

    if relative_path[0] == "/":
        relative_path = relative_path[1:]

    return "/".join(home_path + relative_path.split("/"))
