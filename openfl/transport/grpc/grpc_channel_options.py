# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


max_metadata_size = 32 * 2**20
max_message_length = 2**30

channel_options = [
    ("grpc.max_metadata_size", max_metadata_size),
    ("grpc.max_send_message_length", max_message_length),
    ("grpc.max_receive_message_length", max_message_length),
]
