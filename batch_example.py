import os
import shutil
import asyncio
from bs4 import BeautifulSoup
from translate import translation_pipeline


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def open_as_soup(path: str):
    with open(path, 'r') as fin:
        html_in = fin.read()
    return BeautifulSoup(html_in, 'html.parser')


def save_soup(path: str, soup: BeautifulSoup):
    with open(path, 'w') as fout:
        fout.write(str(soup))


async def translation_task(
        path_src: str,
        path_dst: str
):
    print(f'⏩START: {path_src}')
    soup = open_as_soup(path_src)
    result = await translation_pipeline(soup)
    save_soup(path_dst, soup)
    print(
        f'📝RESULT: S|C|F/ALL: {result.count("S")}|{result.count("C")}|{result.count("F")}/'
        f'{len(result)} | saved as: {path_dst}'
    )


async def main(
        root_src: str,
        root_dst: str
):
    tasks = []
    ensure_dir(root_dst)
    for (dir_src, dir_names, file_names) in os.walk(root_src):
        dir_dst = dir_src.replace(root_src, root_dst)
        ensure_dir(dir_dst)
        for file in file_names:
            path_src = str(os.path.join(dir_src, file))
            path_dst = str(os.path.join(dir_dst, file))
            if os.path.isfile(path_dst):
                continue
            f_name, f_extension = os.path.splitext(file)

            if f_extension == '.html':
                tasks.append(translation_task(path_src, path_dst))
            else:
                shutil.copy(path_src, path_dst)
    print(f'🔰 === STARTING {len(tasks)} TASKS === ')
    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in task_results:
        print(r)

if __name__ == '__main__':
    ori_docs_root = '/root/directory/of/htmls/to/be/translated'
    translated_root = '/root/directory/for/translated/htmls'
    asyncio.run(main(ori_docs_root, translated_root))
