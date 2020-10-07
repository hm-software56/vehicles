from flask import Flask, render_template, Response, request, session
from flask_bootstrap import Bootstrap
import cv2, pafy
from camera import VideoCamera
import imageio
import os, sys
from time import gmtime, strftime, localtime

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.secret_key = 'we213eer341321321eewre123'
webcam_id = 'cars.avi'


@app.route('/', methods=['get', 'post'])
def index():
    session['time'] = 0;
    if request.form.get('url'):
        session['url'] = request.form.get('url')
    return render_template('index.html')


def gen(camera, url):
    url = url
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    camera.video = cv2.VideoCapture(best.url)
    # camera.video = cv2.VideoCapture(webcam_id)
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(gen(VideoCamera(), session['url']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/redovideo', methods=['GET', 'POST'])
def redovideo():
    VideoCamera().__del__()
    if request.args.get('name'):
        session['redo_name'] = request.args.get('name')
    # path = os.listdir(os.path.join('static', 'video'))
    """path = [file for file in os.listdir(os.path.join('static', 'video')) if file.endswith('.avi')]
    try:
        for filename in path:
            convertFile(os.path.join('static', 'video', filename), '.mp4')
            os.remove(os.path.join('static', 'video', filename))
    except:
        print("Can't delete file")"""

    path = [file for file in os.listdir(os.path.join('static', 'video')) if file.endswith('.mp4')]
    path.sort(reverse=True)
    return render_template('redo_video.html', list_data=path)


@app.route('/video_redo', methods=['GET'])
def video_redo():
    return Response(genredovideo(VideoCamera(), session['redo_name']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def genredovideo(camera, redo_name):
    camera.video = cv2.VideoCapture(os.path.join('static', 'video', redo_name))
    while True:
        try:
            frame = camera.get_frame_redo()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except:
            camera.__del__()


@app.route('/convert', methods=['GET'])
def convert():
    path = [file for file in os.listdir(os.path.join('static', 'video')) if file.endswith('.avi')]
    path.sort(reverse=True)
    try:
        i = 0
        for filename in path:
            i = i + 1
            convertFile(os.path.join('static', 'video', filename), '.mp4')
            os.remove(os.path.join('static', 'video', filename))

    except:
        print("Can't delete file")
    return 'done'


def convertFile(inputpath, targetFormat):
    outputpath = os.path.splitext(inputpath)[0] + targetFormat
    print("converting\r\n\t{0}\r\nto\r\n\t{1}".format(inputpath, outputpath))

    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)
    for i, im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    print("\r\nFinalizing...")
    writer.close()
    print("Done.")


if __name__ == '__main__':
    # app.run()
    app.run(debug=True, host='192.168.100.247', port='2020')
