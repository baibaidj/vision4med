import cv2
import os
import os.path as osp
import numpy as np
import nibabel as nb
import torch
import torch.nn.functional as F

#
def mkdir(path):
    # credit goes to YY
    # 判别路径不为空
    path = str(path)
    if path != '':
        # 去除首空格
        path = path.rstrip(' \t\r\n\0')
        if '~' in path:
            path = os.path.expanduser(path)
        # 判别是否存在路径,如果不存在则创建
        if not os.path.exists(path):
            print('=> creating {}'.format(path))
            os.makedirs(path)


def load_image_nii(img_nii_fp, verbose = True):
    nii_obj = nb.load(img_nii_fp)
    affine_matrix = nii_obj.affine
    x_col_sign = int(np.sign(-1 * affine_matrix[0,0]))
    y_row_sign = int(np.sign(-1 * affine_matrix[1,1]))
    if verbose: print('x y sign', x_col_sign, y_row_sign)
    if verbose: print('affine matrix\n', affine_matrix)
    image_3d = np.swapaxes(nii_obj.get_data(), 2, 0)

    image_3d = image_3d[:, ::y_row_sign, ::x_col_sign]
    spacing_list = nii_obj.header.get_zooms()[::-1]
    return image_3d, affine_matrix


class SpatialNormer():
    """
    given old-spacing and new-spacing, 
    transform old shape to new shape
    the dimensions for spacing and shape should be the same
    default is z, y, x; or axial, saggital, coronal 
    
    """
    def __init__(self, affine_matrix, new_spacing, verbose = False):
        assert affine_matrix.shape == (4,4)
        assert len(new_spacing) ==3

        self.affine_matrix = affine_matrix
        self.old_spacing = [affine_matrix[2-i, 2-i] for i in range(3)] # z, y, x
        self.new_spacing = [new_spacing[a] * np.sign(self.old_spacing[a]) for a in range(3)]

        self.verbose = verbose
        if self.verbose: print('\n\nold affine\n', affine_matrix)

    @property
    def x_col_sign(self): return int(-1 * np.sign(self.affine_matrix[0,0]))

    @property
    def y_row_sign(self): return int(-1 * np.sign(self.affine_matrix[1,1]))

    @property
    def z_slice_sign(self): return int(np.sign(self.affine_matrix[2,2]))

    @property
    def new_affine(self):
        new_affine_matrix = np.copy(self.affine_matrix)
        for i in range(3):
            new_affine_matrix[2-i, 2-i] = self.new_spacing[i]
        
        if self.verbose: print('new affine\n', new_affine_matrix)
        return new_affine_matrix

    @property
    def trans_ratio(self):
        ratio = [ abs(self.old_spacing[a]/self.new_spacing[a]) for a 
                 in range(3)]
        if self.verbose: print('trans ratio\n', ratio)
        return ratio
    
    def new_shape(self, old_shape):
        new_shape = [int(old_shape[a] * self.trans_ratio[a]) for a in range(3)]
        if self.verbose: print('new shape\n', new_shape)
        return new_shape

    def transform(self, input_image, mode = 'nearest'):
        if self.verbose: print_tensor('input\n', input_image)
        try:
            input_tensor = torch.from_numpy(input_image.copy())
        except TypeError:
            input_image = np.array(input_image, dtype = np.uint8)
            input_tensor = torch.from_numpy(input_image.copy())
            
        np_dtype = input_image.dtype 
        img_d, img_h, img_w = input_image.shape
        input_tensor = input_tensor.view(1, 1, img_d, img_h, img_w).float()
        resize_tensor = F.interpolate(input_tensor, self.new_shape(input_image.shape), mode=mode).data[0, 0]
        
        if self.verbose: print_tensor('post trans\n', resize_tensor)
        resize_tensor = np.array(resize_tensor, dtype = np_dtype)
        return resize_tensor

print_tensor = lambda name, x : print(name, x.shape, x.dtype, x.min(), x.max())


def array2nii(mask_3d, store_root, file_name, affine_matrix = None, nii_obj= None,
              is_compress = True):
    """
    将输入图像存储为nii
    输入维度是z, r=y, c=x
    输出维度是x=c, y=r, z
    :param mask_3d:
    :param store_root:
    :param file_name:
    :param nii_obj:
    :return:
    """

    extension = 'nii.gz' if is_compress else 'nii'
    mask_3d = np.swapaxes(mask_3d, 0, 2)
    # mask_3d = mask_3d[::-1,::-1,:] # be cautious to uncomment this line
    store_path = osp.join(store_root, '.'.join([file_name, extension]))
    if nii_obj is None:
        if affine_matrix is None: affine_matrix = np.eye(4) 
        # print(affine_matrix)
        nb_ojb = nb.Nifti1Image(mask_3d, affine_matrix)
    else:
        nb_ojb = nb.Nifti1Image(mask_3d, nii_obj.affine, nii_obj.header)

    nb.save(nb_ojb, store_path)
    # print(' done saving nii to ', store_path



def grid2world4slice(trans_matrix, ipp, reverse_xy_spacing = True):
    ipp[:2] = [ipp[i] * (-1 if reverse_xy_spacing else 1) for i in range(2)]
    trans_matrix[:3, 3] = ipp
    return trans_matrix


def grid2world_matrix(spacing, IOP = (1, 0, 0, 0, 1, 0), reverse_xy_spacing = True):
    """
    Nibabel images always use RAS+ output coordinates, where 
    Right, anterior and superior direction in the patient's perspective is regarded as positive.
    reference: https://nipy.org/nibabel/coordinate_systems.html
    

    If Anatomical Orientation Type (0010,2210) is absent or has a value of BIPED, 
    the x-axis is increasing to the left hand side of the patient. 
    The y-axis is increasing to the posterior side of the patient. 
    The z-axis is increasing toward the head of the patient.
    This is LPS+ coordinates system. 
    reference: https://dicom.innolitics.com/ciods/ct-image/general-series/00102210

    Normally, 0010,2210 tag is absent in the dicom of human subjects, 
    which means that most dicom uses LPS+ coordinates system. 
    When conversion to nifti, we need to transform LPS+ to RAS+ 
    by setting the pixel spacing on axial/xy plane to negative. 


    计算像素坐标和物理坐标的变换矩阵
    :param IOP:患者的体位朝向信息，ImageOrientationPatient, 注意和IPP区别
    :param IPP_start:整套图头张slice的物理坐标，IPP
    :param spacing: spacing_list three elements for z, y, x
    :return: 4*4的转换矩阵
    """
    if IOP is None:
        IOP = [1, 0, 0, 0, 1, 0]
    IOP = [float(i) for i in IOP]
    trans_matrix = np.zeros((4, 4), dtype=np.float32)
    trans_matrix[-1, -1] = 1
    x_axis_direction_cosine = np.array(IOP[:3]) * spacing[2] * (-1 if reverse_xy_spacing else 1)
    y_axis_direction_cosine = np.array(IOP[3:]) * spacing[1] * (-1 if reverse_xy_spacing else 1)

    z_axis_direction_cosine = np.array([IOP[1] * IOP[5] - IOP[2] * IOP[4], # 
                                        IOP[2] * IOP[3] - IOP[0] * IOP[5],
                                        IOP[0] * IOP[4] - IOP[1] * IOP[3]]) * spacing[0]

    trans_matrix[0, :3] = x_axis_direction_cosine
    trans_matrix[1, :3] = y_axis_direction_cosine
    trans_matrix[2, :3] = z_axis_direction_cosine

    return trans_matrix

def write_array2png_dl(img_2d, save_dir, file_name, shift_quant = 32768):
    """ based on deep lesion convention
        the img_2d should be in int16
    """
    img_2d += shift_quant
    png_array_16 = np.array(img_2d, dtype=np.uint16)
    cv2.imwrite(os.path.join(save_dir, file_name), png_array_16)  # b, g, r

def save_image(save_path, image_tensors, 
                img_pos = (-250.0, -250.0, -48.0),
                img_orient = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                spacing_list = None, save_formats = ('npz', 'nii', 'png')):
    """
    保存图像为image.npz
    image_tensors 和 spacing_list的输入维度顺序是z, r=y, c=x
    """
    mkdir(save_path)
    affine_matrix = grid2world_matrix(spacing_list, img_orient)
    affine_matrix = grid2world4slice(affine_matrix, img_pos)
    x_col_sign = int(np.sign(img_orient[0]))
    y_row_sign = int(np.sign(img_orient[-2]))
    print('x y sign', x_col_sign, y_row_sign)
    print('affine matrix\n', affine_matrix)
    assert type(save_formats) in (list, tuple)
    # os.makedirs(save_path)
    # print('tensor dtype', image_tensors.dtype)
    for ext in save_formats:
        if 'nii' in ext:
            array2nii(image_tensors, save_path, 'image', affine_matrix)
        elif ext == 'npz':
            np.savez_compressed(save_path + '/image.npz', data=image_tensors)
        elif ext == 'png':
            for idx in range(image_tensors.shape[0]):
                
                write_array2png_dl(image_tensors[idx, ::y_row_sign, ::x_col_sign], save_path, '%03d.png' %idx)
                # write_array2png(image_tensors[idx, :, :], save_path, '%03d.png' %idx, spacing_list)
                # cv2.imwrite(save_path + '/%03d.' % idx + 'png', image_tensors[idx, :, :])
        else:
            print('save formats only support npz, nii and png but give %s' %save_formats) 

# usage:
# image_tensor, affine_matrix = load_image_nii(img_nii_fp, verbose = False)
# mask_tensor, affine_matrix = load_image_nii(gt_nii_fp, verbose = False)

# spatial_normer = SpatialNormer(affine_matrix, new_spacing, verbose=False)
# new_image_tensor = spatial_normer.transform(image_tensor, mode = 'trilinear')
# new_mask_tensor = spatial_normer.transform(mask_tensor, mode = 'nearest')
# x_col_sign, y_row_sign = spatial_normer.x_col_sign, spatial_normer.y_row_sign
# # print('sign, x%s y%s' %(x_col_sign, y_row_sign))
# array2nii(new_image_tensor[:,::y_row_sign, ::x_col_sign], img_dir_abs, new_img_fn, spatial_normer.new_affine)
# array2nii(new_mask_tensor[:,::y_row_sign, ::x_col_sign], gt_dir_abs, new_gt_fn, spatial_normer.new_affine)
