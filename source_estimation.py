
import os
import numpy as np
from mne import setup_source_space, write_source_spaces, read_source_spaces
from mne import make_bem_model, make_bem_solution
from mne import write_bem_surfaces, write_bem_solution, read_bem_surfaces, read_bem_solution
from mne import read_forward_solution, write_forward_solution
from mne import io
from mne import make_forward_solution, compute_covariance
from mne import pick_types, find_events, Epochs
from mne.minimum_norm import make_inverse_operator, read_inverse_operator, write_inverse_operator
from mne import spatio_temporal_src_connectivity, spatial_src_connectivity

def source_setup(subject, spacing='ico4', spatio_temporal=False):

    # SOURCE FILE
    srcfile = '/home/francesco/data/Imagination_freesurfer/%s/src/%s-%s-src.fif' % (subject, subject, spacing)
    if not os.path.isfile(srcfile):
        print('Creating a source space ...')
        src = setup_source_space(subject, spacing=spacing)
        print('Writing source space to file ...')
        write_source_spaces(srcfile, src)

    else:
        print('Loading a source space ...')
        src = read_source_spaces(srcfile)

    if spatio_temporal == True:
        connfile = '/home/francesco/data/Imagination_freesurfer/%s/src/%s-spatio-temporal-conn.npy' % (subject, subject)
        if not os.path.isfile(connfile):
            connectivity = spatio_temporal_src_connectivity(src, n_times=40, dist=None)
            mdict = {}
            mdict['connectivity_matrix'] = connectivity
            mdict['ntimes'] = 40
            mdict['dist'] = 'immediate neighbors'
            np.save(connfile, mdict)
    else:
        connfile = '/home/francesco/data/Imagination_freesurfer/%s/src/%s-spatial-conn.npz' % (subject, subject)
        if not os.path.isfile(connfile):
            connectivity = spatial_src_connectivity(src, dist=None)
            mdict = {}
            mdict['connectivity_matrix'] = connectivity
            mdict['dist'] = 'immediate neighbors (default)'
            np.save(connfile, mdict)

    # BEM SOLUTION
    linkfiles = ['/home/francesco/data/Imagination_freesurfer/%s/bem/inner_skull.surf' % subject,
                 '/home/francesco/data/Imagination_freesurfer/%s/bem/outer_skull.surf' % subject,
                 '/home/francesco/data/Imagination_freesurfer/%s/bem/outer_skin.surf' % subject]
    bemfiles = ['/home/francesco/data/Imagination_freesurfer/%s/bem/%s-bem.fif' % (subject, subject),
                '/home/francesco/data/Imagination_freesurfer/%s/bem/%s-bem-sol.fif' % (subject, subject)]
    if not [f for f in linkfiles if os.path.isfile(f)]:
        print('Link files do not exist in the BEM folder')
        raise ValueError('Please create link files before going on')
    else:
        if not [f for f in bemfiles if os.path.isfile(f)]:
            print('Creating a BEM model ...')
            model = make_bem_model(subject)
            print('Writing a BEM model to file ...')
            write_bem_surfaces(bemfiles[0], model)
            print('Creating a BEM solution ...')
            bem_sol = make_bem_solution(model)
            print('Writing a BEM solution to file ...')
            write_bem_solution(bemfiles[1], bem_sol)
        else:
            print('Loading the BEM solution ...')
            model = read_bem_surfaces(bemfiles[0])
            bem_sol = read_bem_solution(bemfiles[1])

    runs_id = [1, 2, 3, 4]
    if subject[8:12] == 'ABPD':
        fif_template = 'data/Imagination/maxFiltered_data/ABPD/19850501ABPD_201810181645_2009036_run%s_tsss_phs.fif'
    elif subject[8:12] == 'MRCN':
        fif_template = 'data/Imagination/maxFiltered_data/MRCN/19921010MRCN_201811080915_2009036_run%s_tsss_phs.fif'
    elif subject[8:12] == 'MRBU':
        fif_template = 'data/Imagination/maxFiltered_data/MRBU/19940721MRBU_201811081130_2009036_run%s_tsss_phs.fif'
    filename_fif = fif_template % runs_id[0]
    raw = io.read_raw_fif(filename_fif, preload=False)

    # MEG/MRI COORDINATES ALIGNMENT
    fname_trans = '/home/francesco/data/Imagination_freesurfer/%s/bem/%s-trans.fif' % (subject, subject)
    if not os.path.isfile(fname_trans):
        print('MEG/MRI co-registration file does not exist')
        print('See: https://www.slideshare.net/mne-python/mnepython-coregistration')
        raise ValueError('Please create a co-registration file using mne.gui.coregistration() before going on')
    else:
        print('MEG/MRI coordinate systems are aligned')

    # FORWARD SOLUTION
    fwdfile = '/home/francesco/data/Imagination_freesurfer/%s/bem/%s-fwd.fif' % (subject, subject)
    if not os.path.isfile(fwdfile):
        print('Computing the forward solution ...')
        fwd = make_forward_solution(raw.info, fname_trans, src, bem_sol)
        print('Writing the forward solution to file ...')
        write_forward_solution(fwdfile, fwd)
    else:
        print('Loading the forward solution ...')
        fwd = read_forward_solution(fwdfile)

    # NOISE-COVARIANCE MATRIX
    picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, misc=False, exclude='bads')
    events = find_events(raw, shortest_event=0, min_duration=0.003)
    event_id = {'picture_frame': 254}
    epochs = Epochs(raw, events, event_id, tmin=-0.5, tmax=0.5, picks=picks, baseline=(-0.5, 0.0), preload=True)
    print('Computing the noise-covariance matrix ...')
    noise_cov_reg = compute_covariance(epochs, tmax=0.0, method='auto', rank=None)

    # INVERSE SOLUTION
    invfile = '/home/francesco/data/Imagination_freesurfer/%s/bem/%s-inv.fif' % (subject, subject)
    if not os.path.isfile(invfile):
        print('Calculating the inverse operator ...')
        inv = make_inverse_operator(raw.info, fwd, noise_cov_reg, loose=0.2)
        print('Writing the inverse operator to file ...')
        write_inverse_operator(invfile, inv)
    else:
        print('Loading the inverse operator ...')
        inv = read_inverse_operator(invfile)

    return src, connectivity, bem_sol, fwd, noise_cov_reg, inv

if __name__ == '__main__':
    subject = '19850501ABPD_201807200930'
    src, connectivity, bem_sol, fwd, noise_cov_reg, inv = source_setup(subject)