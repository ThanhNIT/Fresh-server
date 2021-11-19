def get_auth_required_parser(api):
    auth_parser = api.parser()
    auth_parser.add_argument("Authorization", type=str, location='headers', required=False)
    return auth_parser


def get_auth_not_required_parser(api):
    auth_parser = api.parser()
    auth_parser.add_argument("Authorization", type=str, location='headers', required=False)
    return auth_parser
