import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import inspect
import functools


# Log images
def log_input_image(x, opts):
    return tensor2im(x)


def tensor2array(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def tensor2image(x):
    x = x.clone()
    x = ((x + 1) / 2)
    x = x.clamp(0., 1.)
    x = x * 255
    return x


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def tensor2map(var):
    mask = np.argmax(var.data.cpu().numpy(), axis=0)
    colors = get_colors()
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return Image.fromarray(mask_image)


def tensor2sketch(var):
    im = var[0].cpu().detach().numpy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = (im * 255).astype(np.uint8)
    return Image.fromarray(im)


# Visualization utils
def get_colors():
    # currently support up to 19 classes (for the celebs-hq-mask dataset)
    colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
              [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
              [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    return colors


def vis_faces(log_hooks):
    display_count = len(log_hooks)
    fig = plt.figure(figsize=(8, 4 * display_count))
    gs = fig.add_gridspec(display_count, 3)
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        if 'diff_input' in hooks_dict:
            vis_faces_with_id(hooks_dict, fig, gs, i)
        else:
            vis_faces_no_id(hooks_dict, fig, gs, i)
    plt.tight_layout()
    return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'])
    plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
                                                     float(hooks_dict['diff_target'])))
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def deduplicate(x_faces):
    indices = []
    slow, fast = 0, 0
    while fast < x_faces['image_ids'].shape[0]:
        id_slow, id_fast = int(x_faces['image_ids'][slow]), int(x_faces['image_ids'][fast])
        if id_slow == id_fast:
            fast += 1
        else:
            indices.append(slow)
            slow = fast
    indices.append(slow)

    return x_faces['alignment'][indices], x_faces['image_ids'][[indices]]


def vis_faces_no_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'], cmap="gray")
    plt.title('Input')
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output')


def configurable(init_func=None, *, from_config=None):
    """
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.

    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass

            @classmethod
            def from_config(cls, cfg):   # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}

        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite

        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg: {"a: cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass

        a1 = a_func(a=1, b=2)  # regular call
        a2 = a_func(cfg)       # call with a cfg
        a3 = a_func(cfg, b=3, c=4)  # call with extra overwrite

    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class must have a ``from_config`` classmethod which takes `cfg` as
            the first argument.
        from_config (callable): the from_config function in usage 2. It must take `cfg`
            as its first argument.
    """

    if init_func is not None:
        assert (
                inspect.isfunction(init_func)
                and from_config is None
                and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    else:
        if from_config is None:
            return configurable  # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)

            wrapped.from_config = from_config
            return wrapped

        return wrapper


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.

    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    from omegaconf import DictConfig

    if len(args) and isinstance(args[0], (DictConfig)):
        return True
    if isinstance(kwargs.pop("cfg", None), (DictConfig)):
        return True
    # `from_config`'s first argument is forced to be "cfg".
    # So the above check covers all cases.
    return False