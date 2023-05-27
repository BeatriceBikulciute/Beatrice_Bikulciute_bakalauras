import base64
import urllib.parse
from io import BytesIO


def get_graph_url(plot):
    buffer = BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = buffer.read()
    buffer.close()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_url = 'data:image/png;base64,' + urllib.parse.quote(image_base64)
    return image_url
