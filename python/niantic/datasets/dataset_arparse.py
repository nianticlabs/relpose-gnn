from niantic.datasets.graph_structure import GraphStructure


def add_arguments_dataset(parser):
    parser.add_argument(
        '--seq-len', type=int,
        help="Number of nodes in graph."
             " Default: 8", default=8,
    )
    parser.add_argument(
        '--sampling-method', type=str, choices=('IR', 'RAND'),
        help='How to sample to create graph: either image retrieval or random.'
             ' Default: IR.', default='IR'
    )
    parser.add_argument(
        '--cross-connect', type=bool,
        help='Perform cross connection between different sequences for the training-set.'
             ' Default: False.', default=False
    )
    parser.add_argument(
        '--graph-structure', type=str,
        choices=tuple(v for k, v in GraphStructure.__dict__.items() if k[0] != '_'),
        help='Graph connectivity strategy.'
             ' Default: "fc".', default=GraphStructure.FC
    )
    group_sampling_strided = parser.add_argument_group('sampling-strided')
    group_sampling_strided.add_argument(
        '--sampling-period', '--sp', dest='sampling_period', type=int,
        help='Strided sampling of neighbors.'
             ' Default: 5.', default=5
    )
    group_sampling_strided.add_argument(
        '--node-dropout', type=float, default=0.5,
        help='Randomly drop each element of the image-retrieval results with this probability.'
             ' Default: 0.5.')

    group_sampling_distanced = parser.add_argument_group('sampling-distanced')
    # Reference: Page 5 in To Learn or Not to Learn: Visual Localization from Essential Matrices
    # https://arxiv.org/pdf/1908.01293.pdf
    group_sampling_distanced.add_argument(
        '--sampling-min-dist', type=float, choices=(0.05, 0.1, 3.),
        help='Minimum distance to all previously selected neighboring views.'
             ' Should be indoors: 0.05, outdoors: 3.'
             ' Default: 0.05 indoors.', default=0.05
    )

    group_sampling_distanced.add_argument(
        '--sampling-max-dist', type=float, choices=(10., 50.),
        help='Maximum distance to all previously selected neighboring views.'
             ' Should be indoors: 10, outdoors: 50.'
             ' Default: 10. indoors.', default=10.
    )
    return parser
