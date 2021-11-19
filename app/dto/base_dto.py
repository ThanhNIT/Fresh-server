from flask_restx import fields, Model

base = Model('base', {
    'status': fields.Boolean,
    'message': fields.String
})
