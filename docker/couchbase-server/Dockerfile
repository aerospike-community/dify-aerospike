FROM couchbase/server:latest@sha256:448f1f837f7fea6a3815c85d2f428768b913046cb8026f73d1a0ae81dd8d1d97 AS stage_base
# FROM couchbase:latest AS stage_base
COPY init-cbserver.sh /opt/couchbase/init/
RUN chmod +x /opt/couchbase/init/init-cbserver.sh
