import streamlit as st
from PIL import Image
from instagrapi import Client
from pathlib import Path
import requests
import os
import glob
import subprocess
import shutil
import sys
import time
from demo import run

from streamlit_js_eval import streamlit_js_eval

### streamlit style options

streamlit_style = """
         <style>
         @import url("https://fonts.googleapis.com/css2?family=Poppins&display=swap");

         html, body, [class*="css"]  {
         font-family: 'Poppins', sans-serif;
         }
            .custom-style {
                margin-top: 115px;  # 이미지 사이의 간격 조절
            }
            .custom-title {
                font-weight: 900;  # 텍스트를 굵게
            }
            </style>
         """
st.markdown(streamlit_style, unsafe_allow_html=True)

###utils

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def insta_crawling(ID, PW,target="jaeu8021"):
    cl = Client()
    crawl_state.text("Try to access instagram...")
    cl.login(ID, PW)
    user_id = cl.user_id_from_username(target)
    crawl_state.text("Feed searching...")

    medias = cl.user_medias_v1(int(user_id), 9)
    print(len(medias))
    if len(medias)<1:
        crawl_state.markdown(f"There're **No** photos: {target}")
        return

    folder = f"{st.session_state.seed}_test-folder"
    createDirectory(folder)
    
    temp = []
    crawl_state.text(f"Saving Image....({len(temp)})")
    for m in medias:
        try:
            p = photo_download(cl, m.pk, folder)
            temp.append(p)
        except AssertionError:
            pass
        crawl_state.text(f"Saving Image....({len(temp)})")
    crawl_state.text("Crawling finished! ") # + os.path.abspath(p))
    st.session_state.crawled = [*map(Image.open, temp)]

    delete_folder(folder)

def photo_download(c, pk, folder):
    media = c.media_info(pk)
    
    filename = "{username}_{media_pk}".format(
        username=media.user.username, media_pk=pk
    )
    p = os.path.join(folder, filename + '.jpg')
    print("INFO", media.media_type)
    if media.media_type==8:
        response = requests.get(media.resources[0].thumbnail_url,
                            stream=True, timeout=c.request_timeout)
    else:
        response = requests.get(media.thumbnail_url,
                            stream=True, timeout=c.request_timeout)
    response.raise_for_status()
    with open(p, "wb") as f:
        f.write(response.content)

    return p

def rotate_img(img):

    try:
        exif = img._getexif()
    except AttributeError:
        exif = None 

    if exif is not None:
        orientation = exif.get(0x0112, 1)
        if orientation == 2:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 4:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
        print(f'exit : {orientation}')

    return img

def concat_image(files, preview=False):  # test folder 에서 이미지를 받아와서 합해야됨
    print("start concating...")
    def resize_squared_img(img):
        h = img.height
        w = img.width
        if w < h:
            m = (h-w)//2
            return img.crop((0, m, w, m+w)), w
        elif h < w:
            m = (w-h)//2
            return img.crop((m, 0, m+h, h)), h
        return img, h

    images = []
    msize = 1000

    for f in files:
        img = f
        img, m = resize_squared_img(img)
        msize = min(m, msize)
        images.append(img)

    def hconcat_resize_pil(im_list,msize):
        im_list_resize = [im.resize((msize, msize))
                          for im in im_list]
        total_width = msize*len(im_list)
        dst = Image.new('RGB', (total_width, msize))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += msize
        return dst

    def vconcat_pil(im_list,msize):
        total_height = msize*len(im_list)
        dst = Image.new('RGB', (msize*3, total_height))
        pos_y = 0
        for im in im_list:
            dst.paste(im, (0, pos_y))
            pos_y += msize
        return dst

    concat_row = []
    n = len(images)
    if n<1:
        return "NO-images"
    for i in range(0, n, 3):
        if n-i < 3:
            break
        row = hconcat_resize_pil(images[i:i+3],msize)
        concat_row.append(row)
    if not concat_row:
        concat_single_image=images[0]
    else:
        concat_single_image = vconcat_pil(concat_row,msize)

    createDirectory('examples')
    createDirectory('examples/style')
    createDirectory('examples/content')
    if preview:
        return concat_single_image
    concat_single_image.save(f'examples/style/{st.session_state.seed}_concat_image.jpg', 'JPEG')
    return "concat-saved"

def update_progress_bar(progress):
    
    if progress < 0.99:
        bar.progress(progress)
    else:
        bar.progress(progress)
        time.sleep(1)
        bar.empty()
        

def delete_folder(filepath):
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
        # os.rmdir(filepath)
        print("delete")
        return "Remove folder"
    else:
        return "Directory Not Found"

def delete_files(filelist):
    for file in filelist:
        if os.path.exists(file):
            os.remove(file)
    return "Remove All File"

def delete_all_files(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)
        return "Remove All File"
    else:
        return "Directory Not Found"

def get_images(li):
    imgs=[]
    for file in li:
        image = Image.open(file)
        imgs.append(image)
    return imgs

def concating(images):
    print("concat-processing!!!")

    single = concat_image(images)

    st.session_state.images = images
    st.session_state.process_idx = 3

def toggle_imethod():
    st.session_state.imethod=0 if st.session_state.imethod else 1

def reset_directory():
    # delete_folder("examples/content")
    # delete_folder("examples/style")
    delete_folder('examples')
    delete_folder("outputs")

# @st.cache(allow_output_mutation=True)
# def st_init():
#     reset_directory()

### streamlit vars

if 'process_idx' not in st.session_state:
    st.session_state.process_idx = 1
if 'crawled' not in st.session_state:
    st.session_state.crawled=[]
if 'uploaded' not in st.session_state:
    st.session_state.uploaded=[]
if 'images' not in st.session_state:
    st.session_state.images=[]
if 'target' not in st.session_state:
    st.session_state.target=None
if 'ref' not in st.session_state:
    st.session_state.ref=0
if 'seed' not in st.session_state:
    st.session_state.seed=0
if 'imethod' not in st.session_state:
    st.session_state.imethod=0 #default crawling(0), uploading(1)

### stramlit UI

st.image("intersection.png", width = 100)
st.markdown('<h1 class="custom-title">AI Color Grader</h1>', unsafe_allow_html=True)
st.subheader('Find the filter that best fits your Instagram feed!')

# for test
# if st.button("refresh"):
#     reset_directory()

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        target_file = st.file_uploader(label="Choose an image to apply color correction",
                                       type=['jpeg', 'png', 'jpg', 'heic'],
                                       label_visibility='visible',
                                       accept_multiple_files=False)
    
    with col2:
        print(st.session_state.seed)
        option=st.selectbox("Get images for AI to anaylze",
                            ("By Instagram crawling","By Uploading"),
                            on_change=toggle_imethod, index=st.session_state.imethod)
        
        if st.session_state.imethod==0: #crawling
            st.text("Get images for AI to anaylze by Instagram login")
            with st.form("crawling"):
                # insta_id = st.text_input("Put your Instagram ID here!")
                # insta_pwd = st.text_input('Put your Instagram password here!',type='password')
                insta_id = "test1_team8"
                insta_pwd = "test1!!"
            
                # username = st.text_input("Put target Instagram ID here if you want!",placeholder="default:your_id")
                username = st.text_input("Put target Instagram ID here if you want!")
                
                submitted = st.form_submit_button("Submit")
                if submitted:
                    if not username:
                        username=insta_id
                    st.write("Crawling photos from ",username)
                    crawl_state=st.text("...")
                    try:
                        insta_crawling(insta_id, insta_pwd,target=username)
                    except Exception as e:
                        st.write("Crawling Failed...", e)
                    concating(st.session_state.crawled)
                    
        elif st.session_state.imethod==1:
            st.session_state.uploaded = st.file_uploader(label="Choose image(s) for AI to analyze",
                                          type=['jpeg', 'png', 'jpg', 'heic'],
                                          label_visibility='visible',
                                          accept_multiple_files=True)
            if st.button("Process Images", type="primary"):
                concating(map(rotate_img, map(Image.open, st.session_state.uploaded)))
        
with st.container():
    ic1, ic2 = st.columns(2)
    print(st.session_state.process_idx)
    if target_file:
        target = Image.open(target_file)
        # Orientation 수정
        target = rotate_img(target)    
        target = target.convert("RGB")
        if not st.session_state.seed:
            st.session_state.seed=time.time()
        createDirectory('examples')
        createDirectory('examples/content')
        target.save(f'examples/content/{st.session_state.seed}_target.jpg', 'JPEG')
        with ic1:
            st.markdown("**Target image**")
            st.image(target)
    with ic2:
        ref_state=st.markdown("")
        if st.session_state.process_idx > 2:
            if not os.path.exists(f'examples/style/{st.session_state.seed}_concat_image.jpg'):
                st.session_state.process_idx=1
                ref_state.markdown("**Error**: try again getting reference images")
            else:   
                ref=Image.open(f'examples/style/{st.session_state.seed}_concat_image.jpg')
                st.image(ref)


if st.session_state.images:
    if st.session_state.crawled:
        ref_state.markdown("**Reference images from CRAWLING**")
    if st.session_state.uploaded:
        ref_state.markdown("**Reference images from uploading**")

    if st.session_state.process_idx<2:
        st.session_state.process_idx = 2    
        
if st.session_state.process_idx == 3 :
    # Delete __pycache__ directories
    pycache_dirs = glob.glob('**/__pycache__', recursive=True)
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
        except OSError as e:
            st.write(f"Error: {pycache_dir} : {e.strerror}")
    if st.button("Start Transfer", type="primary",disabled= not target_file or not st.session_state.images,help="should need target image and ref images"):   
        directory = 'outputs'

        bar = st.progress(0)
        try:
            run(update_progress_bar, seed=st.session_state.seed)
        except FileNotFoundError as e:
            st.write(e)

        with st.container():
            st.image(f'outputs/{st.session_state.seed}_result.jpg', use_column_width=True)

            # concating(images)
            st.session_state.process_idx = 4
            
            # folder_path = './outputs_list'
            # # make slider
            # selected_image_number = st.slider('', 0, 9, 0)
            # # import image
            # image_path = os.path.join(folder_path, f'{st.session_state.seed}_{selected_image_number}_result.png')
            # image = Image.open(image_path)
            # # show image
            # st.image(image)

            # with open(image_path, "rb") as file:
            #     btn = st.download_button(
            #         label="Download",
            #         data=file,
            #         file_name=f"{selected_image_number}.jpg",
            #         mime="image/png"
            #     )
            #     st.session_state.process_idx == 4:
            

if st.session_state.process_idx == 4:
    with open(f'outputs/{st.session_state.seed}_result.jpg', 'rb') as file:
        button = st.download_button(label = '**Download**', data = file, file_name = "Color_Grading.jpg", mime = 'image/jpg')
        
    if st.button("preview feed",help="show image as it uploaded to instagram"):
        preview = rotate_img(Image.open(f'outputs/{st.session_state.seed}_result.jpg'))
        images=[preview]
        if st.session_state.imethod:
            images+=list(map(rotate_img, map(Image.open, st.session_state.uploaded)))
        else:
            images+=st.session_state.crawled
        cimg=concat_image(images, preview=True)
        
        with st.container():
            c1, c2 = st.columns(2)
            c1.image(preview)    
            c2.image(cimg)


if st.button("Finish app"):
    reset_directory()

    streamlit_js_eval(js_expressions="parent.window.location.reload()")

# 서버가 종료되지 않았다면, netstat -lnp | grep [포트번호] 후, kill -9 [process_id]
if __name__ == "__main__":
    if st.session_state.process_idx < 3:
        reset_directory()
    pass