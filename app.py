
import os
import uuid

import torch
from flask import (
    Flask,
    render_template,
    send_from_directory,
    request,
)
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import NumberRange, InputRequired
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Existing AdaIN model imports
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-secret-key-before-deployment"
)

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR, "static", "uploads"
)

app.config["EXAMPLES_FOLDER"] = os.path.join(
    BASE_DIR, "examples"
)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

Bootstrap(app)


# --------------------------------------------------
# WTForms
# --------------------------------------------------

class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0,
        validators=[
            InputRequired(message="Please select an alpha value."),
            NumberRange(
                min=0.0,
                max=1.0,
                message="Alpha must be between 0 and 1."
            )
        ]
    )

    submit = SubmitField("Transfer Style")


# --------------------------------------------------
# MODEL PATHS AND DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VGG_WEIGHTS = os.path.join(
    BASE_DIR, "vgg_normalised.pth"
)

DECODER_WEIGHTS = os.path.join(
    BASE_DIR, "experiment", "experiment1", "decoder_1.pth"
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

def load_models():
    """Load the pretrained AdaIN encoder and decoder."""

    if not os.path.isfile(VGG_WEIGHTS):
        raise FileNotFoundError(
            f"VGG weights not found: {VGG_WEIGHTS}"
        )

    if not os.path.isfile(DECODER_WEIGHTS):
        raise FileNotFoundError(
            f"Decoder weights not found: {DECODER_WEIGHTS}"
        )

    print(f"Loading models on: {device}")

    model_encoder = VGGEncoder(VGG_WEIGHTS).to(device)
    model_decoder = Decoder().to(device)

    # Load decoder checkpoint.
    try:
        checkpoint = torch.load(
            DECODER_WEIGHTS,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            DECODER_WEIGHTS,
            map_location=device
        )

    # Support common checkpoint formats.
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "decoder" in checkpoint:
            checkpoint = checkpoint["decoder"]

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "The decoder checkpoint does not contain a valid state_dict."
        )

    # Handle checkpoints saved using DataParallel.
    if checkpoint and all(
        key.startswith("module.") for key in checkpoint
    ):
        checkpoint = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint.items()
        }

    # This will report a clear error if the checkpoint architecture differs.
    model_decoder.load_state_dict(checkpoint)

    model_encoder.eval()
    model_decoder.eval()

    print("VGG encoder loaded successfully.")
    print("Decoder loaded successfully.")

    return model_encoder, model_decoder


encoder, decoder = load_models()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def allowed_file(filename):
    """Check whether the uploaded file extension is allowed."""

    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_uploaded_file(file_storage):
    """Validate and save an uploaded image with a unique filename."""

    original_name = secure_filename(file_storage.filename)

    if not original_name or not allowed_file(original_name):
        raise ValueError(
            "Please upload a PNG, JPG, or JPEG image."
        )

    extension = original_name.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:
        file_storage.save(file_path)

        # Verify that the uploaded file is a readable image.
        with Image.open(file_path) as image:
            image.verify()

    except (UnidentifiedImageError, OSError, ValueError):
        if os.path.exists(file_path):
            os.remove(file_path)

        raise ValueError(
            "The uploaded file is not a valid image."
        )

    return filename


def existing_upload(filename):
    """Accept only an existing file inside the upload directory."""

    if not filename:
        return None

    filename = secure_filename(filename)

    if not allowed_file(filename):
        return None

    path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if os.path.isfile(path):
        return filename

    return None


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def load_image(image_path):
    """Open an image safely and convert it to RGB."""

    with Image.open(image_path) as image:
        return image.convert("RGB")


# --------------------------------------------------
# ADAIN STYLE TRANSFER
# --------------------------------------------------

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):
    """Generate a stylized image using Adaptive Instance Normalization."""

    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content_tensor = (
        content_transform(content_image)
        .unsqueeze(0)
        .to(device)
    )

    style_tensor = (
        style_transform(style_image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        content_features = encoder(
            content_tensor,
            is_test=True
        )

        style_features = encoder(
            style_tensor,
            is_test=True
        )

        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        # Alpha = 1: full style transfer
        # Alpha = 0: preserve original content features
        stylized_features = (
            alpha * stylized_features
            + (1.0 - alpha) * content_features
        )

        stylized_image = decoder(stylized_features)

    return stylized_image


def save_image(image_tensor, path):
    """Convert the output tensor into a PNG image."""

    image_tensor = image_tensor.detach().cpu().clone()
    image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.clamp(0, 1)

    image = transforms.ToPILImage()(image_tensor)
    image.save(path, format="PNG")


# --------------------------------------------------
# HOME / STYLE TRANSFER ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    # Restore filenames from hidden fields, if available.
    content_filename = existing_upload(form.content_path.data)
    style_filename = existing_upload(form.style_path.data)

    if request.method == "POST":

        if form.validate_on_submit():

            try:
                # Save a newly selected content image.
                if form.content.data and form.content.data.filename:
                    content_filename = save_uploaded_file(
                        form.content.data
                    )

                # Save a newly selected style image.
                if form.style.data and form.style.data.filename:
                    style_filename = save_uploaded_file(
                        form.style.data
                    )

                if not content_filename:
                    raise ValueError(
                        "Please upload a content image."
                    )

                if not style_filename:
                    raise ValueError(
                        "Please upload a style image."
                    )

                alpha = float(form.alpha.data)

                if not 0.0 <= alpha <= 1.0:
                    raise ValueError(
                        "Alpha must be between 0 and 1."
                    )

                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename
                )

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename
                )

                content_image = load_image(content_path)
                style_image = load_image(style_path)

                stylized_tensor = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                # Save each generated result with a unique filename.
                result_filename = (
                    f"stylized_{uuid.uuid4().hex}.png"
                )

                result_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    result_filename
                )

                save_image(stylized_tensor, result_path)

                result_image = result_filename

            except (ValueError, UnidentifiedImageError) as exc:
                error = str(exc)

            except Exception:
                app.logger.exception("Style transfer failed.")

                error = (
                    "Style transfer failed. Please check the uploaded "
                    "images and model checkpoint, then try again."
                )

        else:
            if form.errors:
                error_messages = []

                for field_errors in form.errors.values():
                    error_messages.extend(field_errors)

                error = " ".join(error_messages)

            else:
                error = (
                    "The form could not be submitted. "
                    "Please refresh the page and try again."
                )

    # Keep the selected filenames in hidden fields for subsequent posts.
    form.content_path.data = content_filename or ""
    form.style_path.data = style_filename or ""

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# --------------------------------------------------
# SERVE UPLOADED / GENERATED IMAGES
# --------------------------------------------------

@app.route("/uploads/<path:filename>")
def send_image(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# --------------------------------------------------
# SERVE EXAMPLE IMAGES
# --------------------------------------------------

@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(
        app.config["EXAMPLES_FOLDER"],
        filename
    )


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return (
        "Image too large. Please upload an image smaller than 16 MB.",
        413
    )


# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

