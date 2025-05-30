AttributeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/crypto-predicator/app.py", line 212, in <module>
    st.image(article.get("urlToImage"), width=600)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/metrics_util.py", line 444, in wrapped_func
    result = non_optional_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/image.py", line 181, in image
    marshall_images(
    ~~~~~~~~~~~~~~~^
        self.dg._get_delta_path_str(),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        output_format,
        ^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/image_utils.py", line 442, in marshall_images
    proto_img.url = image_to_url(
                    ~~~~~~~~~~~~^
        image, width, clamp, channels, output_format, image_id
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/image_utils.py", line 333, in image_to_url
    image_format = _validate_image_format_string(image_data, output_format)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/image_utils.py", line 116, in _validate_image_format_string
    if _image_is_gif(pil_image):
       ~~~~~~~~~~~~~^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/image_utils.py", line 88, in _image_is_gif
    return image.format == "GIF"
           ^^^^^^^^^^^^
