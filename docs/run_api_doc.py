import os
import subprocess

current_dir = os.path.dirname(os.path.realpath(__file__))
out_dir = os.path.join(current_dir, "source")
root_dir = os.path.join(current_dir, os.pardir)


def create_api_doc():
    print("Creating API doc.")

    libs = ["thermography", "gui"]
    for lib in libs:
        print("Parsing {} package".format(lib))
        lib_path = os.path.abspath(os.path.join(root_dir, lib))

        command = "sphinx-apidoc -f -T -M -o {} {}".format(out_dir, lib_path)
        print("Command: {}".format(command))

        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, cwd=current_dir)
        output, error = process.communicate()

        print(output.decode("utf-8"))


def readme_to_rst():
    print("Converting README.md to README.rst file.")

    in_readme_path = os.path.abspath(os.path.join(root_dir, "README.md"))
    out_readme_path = os.path.join(out_dir, "README.rst")

    with open(in_readme_path, "r") as file:
        readme_lines = file.readlines()
        title_lines = [line for line in readme_lines if "# " in line]
        if len(title_lines)==0:
            title_line_index = -1
        else:
            title_line = sorted(title_lines, key=lambda x: x.count("#"))[0]
            title_line_index = title_lines.index(title_line)

    tmp_file = os.path.join(out_dir, "tmp_readme.md")
    with open(tmp_file, "w") as file:
        for i, line in enumerate(readme_lines):
            if i == title_line_index:
                continue
            if (".jpg" in line or ".JPG" in line or ".PNG" in line or ".png" in line or ".gif" in line) and "https" not in line:
                line = line.replace("?raw=true", "") # Remove image tag used in MD file.
                line = line.replace("./docs/", "")
                line = line.replace("docs/", "")
            if "lang=" in line:
                line = line.replace("lang=", "")

            if "# Documentation" in line:
                break
            file.write(line)


    if os.path.exists(out_readme_path):
        os.remove(out_readme_path)
    command = "pandoc --from=markdown --to=rst --output={} {}".format(out_readme_path, tmp_file)

    print("Command: {}".format(command))

    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, cwd=current_dir)
    output, error = process.communicate()

    print(output.decode("utf-8"))
    os.remove(tmp_file)


if __name__ == '__main__':
    readme_to_rst()

    create_api_doc()
