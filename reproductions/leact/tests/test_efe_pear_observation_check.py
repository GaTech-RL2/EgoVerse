from PIL import Image
import pytest

from reproductions.leact.cluster.efe_pear_observation_check import validate_frames


def test_blank_rendering_cannot_qualify_as_a_camera():
    with pytest.raises(ValueError,match='blank'):
        validate_frames([Image.new('RGB',(64,48),'black')],['front'])


def test_rgb_evidence_matches_each_named_native_sensor():
    image=Image.new('RGB',(64,48),'black');image.paste('white',(0,0,32,48))
    result=validate_frames([image,image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)],['left','right'])
    assert result['left']['pixels_sha256']!=result['right']['pixels_sha256']
    assert result['left']['width']==64 and result['left']['height']==48
    assert min(result['left']['channel_stddev'])>1


def test_missing_camera_is_not_a_valid_observation():
    with pytest.raises(ValueError,match='missing'):
        validate_frames([Image.new('RGB',(64,48))],['left','right'])
