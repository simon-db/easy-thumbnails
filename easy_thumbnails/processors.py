import re
import six

try:
    from PIL import Image, ImageChops, ImageFilter
except ImportError:
    import Image
    import ImageChops
    import ImageFilter
from easy_thumbnails import utils


def _compare_entropy(start_slice, end_slice, slice, difference):
    """
    Calculate the entropy of two slices (from the start and end of an axis),
    returning a tuple containing the amount that should be added to the start
    and removed from the end of the axis.

    """
    start_entropy = utils.image_entropy(start_slice)
    end_entropy = utils.image_entropy(end_slice)
    if end_entropy and abs(start_entropy / end_entropy - 1) < 0.01:
        # Less than 1% difference, remove from both sides.
        if difference >= slice * 2:
            return slice, slice
        half_slice = slice // 2
        return half_slice, slice - half_slice
    if start_entropy > end_entropy:
        return 0, slice
    else:
        return slice, 0


def colorspace(im, bw=False, replace_alpha=False, **kwargs):
    """
    Convert images to the correct color space.

    A passive option (i.e. always processed) of this method is that all images
    (unless grayscale) are converted to RGB colorspace.

    This processor should be listed before :func:`scale_and_crop` so palette is
    changed before the image is resized.

    bw
        Make the thumbnail grayscale (not really just black & white).

    replace_alpha
        Replace any transparency layer with a solid color. For example,
        ``replace_alpha='#fff'`` would replace the transparency layer with
        white.

    """
    is_transparent = utils.is_transparent(im)
    if bw:
        if im.mode in ('L', 'LA'):
            return im
        if is_transparent:
            return im.convert('LA')
        else:
            return im.convert('L')

    if im.mode in ('L', 'RGB'):
        return im

    if is_transparent:
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        if not replace_alpha:
            return im
        base = Image.new('RGBA', im.size, replace_alpha)
        base.paste(im, mask=im)
        im = base

    return im.convert('RGB')


def autocrop(im, autocrop=False, **kwargs):
    """
    Remove any unnecessary whitespace from the edges of the source image.

    This processor should be listed before :func:`scale_and_crop` so the
    whitespace is removed from the source image before it is resized.

    autocrop
        Activates the autocrop method for this image.

    """
    if autocrop:
        bw = im.convert('1')
        bw = bw.filter(ImageFilter.MedianFilter)
        # White background.
        bg = Image.new('1', im.size, 255)
        diff = ImageChops.difference(bw, bg)
        bbox = diff.getbbox()
        if bbox:
            im = im.crop(bbox)
    return im


def scale_and_crop(im, size, crop=False, upscale=False, **kwargs):
    """
    Handle scaling and cropping the source image.

    Images can be scaled / cropped against a single dimension by using zero
    as the placeholder in the size. For example, ``size=(100, 0)`` will cause
    the image to be resized to 100 pixels wide, keeping the aspect ratio of
    the source image.

    crop
        Crop the source image height or width to exactly match the requested
        thumbnail size (the default is to proportionally resize the source
        image to fit within the requested thumbnail size).

        By default, the image is centered before being cropped. To crop from
        the edges, pass a comma separated string containing the ``x`` and ``y``
        percentage offsets (negative values go from the right/bottom). Some
        examples follow:

        * ``crop="0,0"`` will crop from the left and top edges.

        * ``crop="-10,-0"`` will crop from the right edge (with a 10% offset)
          and the bottom edge.

        * ``crop=",0"`` will keep the default behavior for the x axis
          (horizontally centering the image) and crop from the top edge.

        The image can also be "smart cropped" by using ``crop="smart"``. The
        image is incrementally cropped down to the requested size by removing
        slices from edges with the least entropy.

        Finally, you can use ``crop="scale"`` to simply scale the image so that
        at least one dimension fits within the size dimensions given (you may
        want to use the upscale option too).

    upscale
        Allow upscaling of the source image during scaling.

    """
    source_x, source_y = [float(v) for v in im.size]
    target_x, target_y = [float(v) for v in size]

    if crop or not target_x or not target_y:
        scale = max(target_x / source_x, target_y / source_y)
    else:
        scale = min(target_x / source_x, target_y / source_y)

    # Handle one-dimensional targets.
    if not target_x:
        target_x = source_x * scale
    elif not target_y:
        target_y = source_y * scale

    if scale < 1.0 or (scale > 1.0 and upscale):
        # Resize the image to the target size boundary. Round the scaled
        # boundary sizes to avoid floating point errors.
        im = im.resize((int(round(source_x * scale)),
                        int(round(source_y * scale))),
                       resample=Image.ANTIALIAS)

    if crop:
        # Use integer values now.
        source_x, source_y = im.size
        # Difference between new image size and requested size.
        diff_x = int(source_x - min(source_x, target_x))
        diff_y = int(source_y - min(source_y, target_y))
        if diff_x or diff_y:
            # Center cropping (default).
            halfdiff_x, halfdiff_y = diff_x // 2, diff_y // 2
            box = [halfdiff_x, halfdiff_y,
                   min(source_x, int(target_x) + halfdiff_x),
                   min(source_y, int(target_y) + halfdiff_y)]
            # See if an edge cropping argument was provided.
            edge_crop = (isinstance(crop, six.string_types) and
                         re.match(r'(?:(-?)(\d+))?,(?:(-?)(\d+))?$', crop))
            if edge_crop and filter(None, edge_crop.groups()):
                x_right, x_crop, y_bottom, y_crop = edge_crop.groups()
                if x_crop:
                    offset = min(int(target_x) * int(x_crop) // 100, diff_x)
                    if x_right:
                        box[0] = diff_x - offset
                        box[2] = source_x - offset
                    else:
                        box[0] = offset
                        box[2] = source_x - (diff_x - offset)
                if y_crop:
                    offset = min(int(target_y) * int(y_crop) // 100, diff_y)
                    if y_bottom:
                        box[1] = diff_y - offset
                        box[3] = source_y - offset
                    else:
                        box[1] = offset
                        box[3] = source_y - (diff_y - offset)
            # See if the image should be "smart cropped".
            elif crop == 'smart':
                left = top = 0
                right, bottom = source_x, source_y
                while diff_x:
                    slice = min(diff_x, max(diff_x // 5, 10))
                    start = im.crop((left, 0, left + slice, source_y))
                    end = im.crop((right - slice, 0, right, source_y))
                    add, remove = _compare_entropy(start, end, slice, diff_x)
                    left += add
                    right -= remove
                    diff_x = diff_x - add - remove
                while diff_y:
                    slice = min(diff_y, max(diff_y // 5, 10))
                    start = im.crop((0, top, source_x, top + slice))
                    end = im.crop((0, bottom - slice, source_x, bottom))
                    add, remove = _compare_entropy(start, end, slice, diff_y)
                    top += add
                    bottom -= remove
                    diff_y = diff_y - add - remove
                box = (left, top, right, bottom)
            # Finally, crop the image!
            if crop != 'scale':
                im = im.crop(box)
    return im


def filters(im, detail=False, sharpen=False, **kwargs):
    """
    Pass the source image through post-processing filters.

    sharpen
        Sharpen the thumbnail image (using the PIL sharpen filter)

    detail
        Add detail to the image, like a mild *sharpen* (using the PIL
        ``detail`` filter).

    """
    if detail:
        im = im.filter(ImageFilter.DETAIL)
    if sharpen:
        im = im.filter(ImageFilter.SHARPEN)
    return im
    
# Custom processors

def custom_scale(im, size, **kwargs):
    source_x, source_y = [float(v) for v in im.size]
    target_x, target_y = [float(v) for v in size]

    if not target_x or not target_y:
        scale = max(target_x / source_x, target_y / source_y)
    else:
        scale = min(target_x / source_x, target_y / source_y)

    if scale < 1.0 :
        # Resize the image to the target size boundary. Round the scaled
        # boundary sizes to avoid floating point errors.
        im = im.resize((int(round(source_x * scale)),
                        int(round(source_y * scale))),
                       resample=Image.ANTIALIAS)
    return im

def prefect_scale(im, size, prefect=False, **kwargs):
    if prefect:
        im = custom_scale(im, (size[0], 0))
        im = custom_scale(im, (0, size[1]))
    return im


def _percent(var):
    """
    Just a simple interface to the _val function with a more meaningful name.
    """
    return _val(var, True)

def _int(var):
    """
    Just a simple interface to the _val function with a more meaningful name.
    """
    return _val(var)

def _val(var, is_percent=False):
    """
    Tries to determine the appropriate value of a particular variable that is
    passed in.  If the value is supposed to be a percentage, a whole integer
    will be sought after and then turned into a floating point number between
    0 and 1.  If the value is supposed to be an integer, the variable is cast
    into an integer.
    """
    try:
        if is_percent:
            var = float(int(var.strip('%')) / 100.0)
        else:
            var = int(var)
    except ValueError:
        raise ValueError('invalid watermark parameter: ' + var)
    return var

def reduce_opacity(img, opacity):
    """
    Returns an image with reduced opacity.
    """
    assert opacity >= 0 and opacity <= 1

    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    else:
        img = img.copy()

    alpha = img.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    img.putalpha(alpha)

    return img

def determine_scale(scale, img, mark):
    """
    Scales an image using a specified ratio or 'F'.  If `scale` is 'F', the
    image is scaled to be as big as possible to fit in `img` without falling off
    the edges.  Returns the scaled `mark`.
    """

    if scale:
        try:
            scale = float(scale)
        except (ValueError, TypeError):
            pass

        if type(scale) in (str, unicode) and scale.lower() == 'f':
            # scale, but preserve the aspect ratio
            scale = min(
                        float(img.size[0]) / mark.size[0],
                        float(img.size[1]) / mark.size[1]
                       )
        elif type(scale) not in (float, int):
            raise ValueError('Invalid scale value "%s"!  Valid values are 1) "F" for ratio-preserving scaling and 2) floating-point numbers and integers greater than 0.' % (scale,))

        # determine the new width and height
        w = int(mark.size[0] * float(scale))
        h = int(mark.size[1] * float(scale))

        # apply the new width and height, and return the new `mark`
        return (w, h)
    else:
        return mark.size

def determine_position(position, img, mark):
    """
    Options:
        TL: top-left
        TR: top-right
        BR: bottom-right
        BL: bottom-left
        C: centered
        R: random
        X%xY%: relative positioning on both the X and Y axes
        X%xY: relative positioning on the X axis and absolute positioning on the
              Y axis
        XxY%: absolute positioning on the X axis and relative positioning on the
              Y axis
        XxY: absolute positioning on both the X and Y axes
    """

    max_left = max(img.size[0] - mark.size[0], 0)
    max_top = max(img.size[1] - mark.size[1], 0)

    if not position:
        position = 'r'

    if isinstance(position, tuple):
        left, top = position
    elif isinstance(position, str) or isinstance(position, unicode):
        position = position.lower()

        # corner positioning
        if position in ['tl', 'tr', 'br', 'bl']:
            if 't' in position:
                top = 0
            elif 'b' in position:
                top = max_top
            if 'l' in position:
                left = 0
            elif 'r' in position:
                left = max_left

        # center positioning
        elif position == 'c':
            left = int(max_left / 2)
            top = int(max_top / 2)

        # random positioning
        elif position == 'r':
            left = random.randint(0, max_left)
            top = random.randint(0, max_top)

        # relative or absolute positioning
        elif 'x' in position:
            left, top = position.split('x')

            if '%' in left:
                left = max_left * _percent(left)
            else:
                left = _int(left)

            if '%' in top:
                top = max_top * _percent(top)
            else:
                top = _int(top)
    return (int(left), int(top))



def watermark(img, mark, position=(0, 0), opacity=1, scale=1.0, tile=False, **kwargs):
    if opacity < 1:
        mark = reduce_opacity(mark, opacity)

    if type(scale) != tuple:
        scale = determine_scale(scale, img, mark)

    mark = mark.resize(scale)
    position = determine_position(position, img, mark)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # make sure we have a tuple for a position now
    assert isinstance(position, tuple), 'Invalid position "%s"!' % position

    # create a transparent layer the size of the image and draw the
    # watermark in that layer.
    layer = Image.new('RGBA', img.size, (0,0,0,0))
    if tile:
        first_y = position[1] % mark.size[1] - mark.size[1]
        first_x = position[0] % mark.size[0] - mark.size[0]

        for y in range(first_y, img.size[1], mark.size[1]):
            for x in range(first_x, img.size[0], mark.size[0]):
                layer.paste(mark, (x, y))
    else:
        layer.paste(mark, position)
    return Image.composite(layer, img, layer)

def add_watermark(im, size, crop=False, mark=False, **kwargs):
    if mark:
        if os.path.exists(mark):
            try:
                wmark = Image.open(mark)
                position=(0,0)
                opacity=1
                scale=1.0

                if 'position' in kwargs:
                    position = kwargs['position']

                if 'opacity' in kwargs:
                    opacity = kwargs['opacity']

                if 'scale' in kwargs:
                    scale = kwargs['scale']

                im = watermark(im, wmark, position, opacity, scale, **kwargs)
            except:
                pass
    return im
