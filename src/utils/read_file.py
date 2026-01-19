def read_file(file_path, *args, **kwargs):
  with open(file_path, 'r', encoding='utf-8', *args, **kwargs) as file:
    content = file.read()
  return content