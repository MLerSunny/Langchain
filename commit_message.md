feat(core): unify config, harden API, fix vector-db drift

* Single Pydantic Settings object (core/settings.py) is the source of truth
* Vector-store choice (chroma/qdrant) now env-controlled and respected everywhere
* FastAPI protected with JWT validation + rate limiting
* Added explicit request/response models
* Improved sentence-aware chunking
* Parameterised fine-tuning hyper-params
* Docker + CI hygiene; added first unit test
