import os
import sys
import urllib.request
import zipfile
import tarfile

# reporthook of `urllib.request.urlretrieve`
def _print_download_progress(count, block_size, total_size):

    """
    :param: count - count of block
    :param: block_size - size of block
    :param: total_size - size of file
    """

    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download complete: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


# download zip or tar file from url to local, and extract it
def maybe_download_and_extract(url, download_dir):

    """
    :param, url: url of file
    :param, download_dir: local dir
    """

    file_name = url.split('/')[-1]
    file_path = os.path.join(download_dir, file_name)

    # if or not downloaded
    if not os.path.exists(file_path):
        # if or not local dir exists, if not, create
        if not os.path.exists(download_dir):
            os.mkdir(download_dir)

        # download url file inside it, and return the file_path
        file_path, _ = urllib.request.urlretrieve(url = url,
                                                  # 这里的 filename 是文件全路径包括文件名
                                                  # 意思是下载和重命名一条龙服务
                                                  # 注意下载的是 tar 包名, 解压之后才是文件夹名
                                                  filename = file_path,
                                                  # 这个回调函数必须是三个参数 blk_count, blk_size, total_size
                                                  # 并且给出打印, 打印语句必须是以 "\r- " 打头
                                                  reporthook = _print_download_progress)

        print()
        print("Download Completed")

        # extract file to local dir
        if file_path.endswith('.zip'):
            # 这里 file 是压缩文件全路径, download_dir 是解压缩的目标文件夹.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)

        elif file_path.endswith(('.tar.gz', 'tgz')):
            # 这里 name 是压缩文件全路径, download_dir 是解压缩的目标文件夹.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

    else:
        print(" url file seem has been downloaded!")


# 可以通过如下语句测试:
# download_and_extract("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "yiddish_dir")
