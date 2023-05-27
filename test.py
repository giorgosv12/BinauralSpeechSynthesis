"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import numpy as np
import torch as th
import torchaudio as ta

from src.models import BinauralNetwork


def chunked_forwarding(net, mono, view):
    '''
    binauralized the mono input given the view
    :param net: binauralization network
    :param mono: 1 x T tensor containing the mono audio signal
    :param view: 7 x K tensor containing the view as 3D positions and quaternions for orientation (K = T / 400)
    :return: 2 x T tensor containing binauralized audio signal
    '''
    net.eval()
    # mono, view = mono.cuda(), view.cuda()

    chunk_size = 480000  # forward in chunks of 10s
    rec_field = net.receptive_field() + 1000  # add 1000 samples as "safe bet" since warping has undefined rec. field
    rec_field -= rec_field % 400  # make sure rec_field is a multiple of 400 to match audio and view frequencies
    chunks = [
        {
            "mono": mono[:, max(0, i - rec_field):i + chunk_size],
            "view": view[:, max(0, i - rec_field) // 400:(i + chunk_size) // 400]
        }
        for i in range(0, mono.shape[-1], chunk_size)
    ]

    for i, chunk in enumerate(chunks):
        with th.no_grad():
            mono = chunk["mono"].unsqueeze(0)
            view = chunk["view"].unsqueeze(0)
            binaural = net(mono, view)["output"].squeeze(0)
            if i > 0:
                binaural = binaural[:, -(mono.shape[-1] - rec_field):]
            chunk["binaural"] = binaural

    binaural = th.cat([chunk["binaural"] for chunk in chunks], dim=-1)
    binaural = th.clamp(binaural, min=-1, max=1).cpu()

    return binaural


def predict(input_file_wav_path, input_positions_filepath):

    model_filepath = "model/binaural_network_1block.net"
    output_dirpath = "outputs/"

    if "1" in model_filepath:
        num_blocks = 1
    elif "3" in model_filepath:
        num_blocks = 3
    else:
        raise Exception("Could not define the number of blocks for the model")

    os.makedirs(output_dirpath, exist_ok=True)

    # binauralized and evaluate test sequence for the eight subjects and the validation sequence
    test_sequences = [f"subject{i+1}" for i in range(8)] + ["validation_sequence"]

    # initialize network
    net = BinauralNetwork(view_dim=7,
                          warpnet_layers=4,
                          warpnet_channels=64,
                          wavenet_blocks=num_blocks,
                          layers_per_block=10,
                          wavenet_channels=64
                          )
    net.load_from_file(model_filepath)

    # load mono input and view conditioning
    mono, sr = ta.load(input_file_wav_path)
    view = np.loadtxt(input_positions_filepath).transpose().astype(np.float32)
    view = th.from_numpy(view)

    # sanity checks
    if not sr == 48000:
        raise Exception(f"sampling rate is expected to be 48000 but is {sr}.")
    if not view.shape[-1] * 400 == mono.shape[-1]:
        raise Exception(f"mono signal is expected to have 400x the length of the position/orientation sequence.")

    output_filename = "{}_binaural.wav".format(os.path.split(input_file_wav_path)[0])

    # binauralize and save output
    binaural = chunked_forwarding(net, mono, view)
    ta.save(os.path.join(output_dirpath, output_filename), binaural, sr)


if __name__ == '__main__':

    input_filepath = "inputs/mono.wav"
    input_position_filepath = "inputs/rx_positions.txt"

    predict(input_filepath, input_position_filepath)
