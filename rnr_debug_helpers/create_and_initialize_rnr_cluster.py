import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from os import path

from rnr_debug_helpers.utils.rnr_wrappers import RetrieveAndRankProxy
from rnr_debug_helpers.utils.io_helpers import initialize_logger, load_config

LOGGER = initialize_logger(logging.INFO, path.basename(__file__))
CONFIG = load_config()
# TODO: make the cluster size an optional parameter
_CLUSTER_SIZE = 2


def main(args):
    LOGGER.info('Start Script')
    solr_cluster = RetrieveAndRankProxy(solr_cluster_id=args.cluster_id, cluster_name='TestCluster',
                                        cluster_size=_CLUSTER_SIZE)
    LOGGER.info('Initialized bluemix connection to solr cluster: %s' % solr_cluster.solr_cluster_id)

    solr_cluster.setup_cluster_and_collection(config_id=args.config_id,
                                              config_zip=args.config_path,
                                              collection_id=args.collection_name)

    LOGGER.info('Initialized a document collection: %s (there are %d docs in the collection)' %
                (args.collection_name, solr_cluster.get_num_docs_in_collection(args.collection_name)))

    solr_cluster.upload_documents_to_collection(args.collection_name, args.corpus_path, content_type='application/xml')

    LOGGER.info('Uploading documents completed, now there are %d docs in the collection. WARNING: depending on the '
                'size of the corpus, there may be a latency between upload and the doc count to reflect the newly'
                ' indexed docs.' % solr_cluster.get_num_docs_in_collection(args.collection_name))

    LOGGER.info('Great Success')


if __name__ == '__main__':
    parser = ArgumentParser(prog="python %s)" % path.basename(__file__),
                            description="Helper script to create a RnR cluster and initialize it with a document"
                                        " corpus (i.e. collection)",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--clusterId', dest='cluster_id', help="Optionally re-use an existing cluster by "
                                                                     "specifying an id.  If none is provided,"
                                                                     " a new one gets created",
                        required=False, default=None)
    parser.add_argument('-o', '--configId', dest='config_id', default=None,
                        help="Optionally re-use an existing config ID that has been uploaded to the service to"
                             " be used to setup the collection")
    parser.add_argument('-f', '--cofigPath', dest='config_path', default=None,
                        help="If a config ID was not specified, indicate the file path to a zipped solr config folder"
                             " that will be used to setup the collection")
    parser.add_argument('-n', '--collectionName', dest='collection_name', required=True,
                        help='The name of the collection that you want to create')
    parser.add_argument('-d', '--documentsPath', dest='corpus_path', required=True,
                        help="The file path to the document corpus that will be uploaded to the new collection (must"
                             " be pre-formatted into Solr format: https://cwiki.apache.org/confluence/display/solr/"
                             "Uploading+Data+with+Index+Handlers)")
    parser.add_argument('-x', '--contentType', dest='content_type', choices=['application/json', 'application/xml'],
                        default='application/json',
                        help="indicate the content type that will be set when using the Solr APIs to upload the "
                             "corpus to the collection")
    parser.add_argument('-v', '--verbose', help="Print lots of debugging statements", action="store_const",
                        dest="log_level", const=logging.DEBUG, default=logging.INFO)
    args = parser.parse_args()
    LOGGER = initialize_logger(args.log_level, path.basename(__file__))
    # get on with it
    main(args)
