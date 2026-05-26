--
-- PostgreSQL database dump
--


-- Dumped from database version 16.13 (Ubuntu 16.13-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.13 (Ubuntu 16.13-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: nocase; Type: COLLATION; Schema: public; Owner: -
--

CREATE COLLATION public.nocase (provider = icu, deterministic = false, locale = 'und-u-ks-level2');


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: admin_chat_confirmations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.admin_chat_confirmations (
    id integer NOT NULL,
    chat_id integer NOT NULL,
    action_id text NOT NULL,
    function_name text NOT NULL,
    params text DEFAULT '{}'::text NOT NULL,
    confirm_string text NOT NULL,
    user_id integer,
    status text DEFAULT 'pending'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    expires_at timestamp with time zone DEFAULT (now() + '00:30:00'::interval) NOT NULL
);


--
-- Name: admin_chat_confirmations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.admin_chat_confirmations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: admin_chat_confirmations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.admin_chat_confirmations_id_seq OWNED BY public.admin_chat_confirmations.id;


--
-- Name: admin_chat_messages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.admin_chat_messages (
    id integer NOT NULL,
    chat_id integer NOT NULL,
    role text NOT NULL,
    content text NOT NULL,
    sources text,
    tool_calls text,
    metadata text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: admin_chat_messages_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.admin_chat_messages_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: admin_chat_messages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.admin_chat_messages_id_seq OWNED BY public.admin_chat_messages.id;


--
-- Name: admin_chat_wizards; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.admin_chat_wizards (
    id integer NOT NULL,
    chat_id integer NOT NULL,
    user_id integer,
    function_name text NOT NULL,
    collected_params text DEFAULT '{}'::text NOT NULL,
    status text DEFAULT 'collecting'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: admin_chat_wizards_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.admin_chat_wizards_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: admin_chat_wizards_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.admin_chat_wizards_id_seq OWNED BY public.admin_chat_wizards.id;


--
-- Name: admin_chats; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.admin_chats (
    id integer NOT NULL,
    user_id integer,
    title text DEFAULT 'Neuer Chat'::text NOT NULL,
    status text DEFAULT 'active'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: admin_chats_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.admin_chats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: admin_chats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.admin_chats_id_seq OWNED BY public.admin_chats.id;


--
-- Name: ai_config; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ai_config (
    id integer NOT NULL,
    embedding_url text DEFAULT ''::text,
    embedding_model text DEFAULT ''::text,
    llm_url text DEFAULT ''::text,
    llm_model text DEFAULT ''::text,
    cloud_enabled boolean DEFAULT false,
    cloud_provider text DEFAULT ''::text,
    cloud_api_key text DEFAULT ''::text,
    cloud_model text DEFAULT ''::text,
    cloud_base_url text DEFAULT ''::text,
    prompt_template text DEFAULT ''::text,
    cloud_mode text DEFAULT 'auto'::text,
    anon_enabled boolean DEFAULT true,
    anon_erp_enabled boolean DEFAULT false,
    anon_erp_path text DEFAULT '/etc/kiara/stammdaten'::text,
    anon_ldap_enabled boolean DEFAULT true,
    anon_word_boundaries integer DEFAULT 1,
    anon_min_term_length integer DEFAULT 3,
    anon_excluded_terms text DEFAULT '[]'::text,
    anon_custom_patterns text DEFAULT '[]'::text,
    anon_audit_log integer DEFAULT 1,
    anon_custom_entries text DEFAULT '[]'::text,
    anon_llm_enabled boolean DEFAULT false,
    anon_llm_model text DEFAULT ''::text,
    anon_llm_temperature text DEFAULT '0.1'::text,
    anon_llm_confidence text DEFAULT '0.7'::text,
    anon_llm_timeout integer DEFAULT 10,
    anon_llm_cache_size integer DEFAULT 256,
    anon_llm_num_ctx integer DEFAULT 4096,
    agent_system_prompt text DEFAULT ''::text
);


--
-- Name: api_keys; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.api_keys (
    id integer NOT NULL,
    name text DEFAULT ''::text NOT NULL,
    key_hash text NOT NULL,
    key_prefix text DEFAULT ''::text NOT NULL,
    user_id integer,
    is_active boolean DEFAULT true,
    expires_at timestamp with time zone,
    last_used_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: api_keys_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.api_keys ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.api_keys_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: audit_anonymization; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.audit_anonymization (
    id integer NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    user_id integer,
    session_id text NOT NULL,
    question_hash text NOT NULL,
    entity_count integer DEFAULT 0 NOT NULL,
    entity_types text DEFAULT '{}'::text NOT NULL,
    route text NOT NULL,
    content_class text,
    cloud_provider text,
    cloud_model text,
    response_time_ms integer,
    correlation_id text
);


--
-- Name: audit_anonymization_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.audit_anonymization_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: audit_anonymization_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.audit_anonymization_id_seq OWNED BY public.audit_anonymization.id;


--
-- Name: cache_datasources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cache_datasources (
    datasource_id integer NOT NULL,
    payload_json jsonb NOT NULL,
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: cache_jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cache_jobs (
    job_id integer NOT NULL,
    datasource_id integer DEFAULT 0,
    status text DEFAULT ''::text,
    payload_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: cache_vectordb; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cache_vectordb (
    id integer NOT NULL,
    payload_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT cache_vectordb_id_check CHECK ((id = 1))
);


--
-- Name: chat_messages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chat_messages (
    id integer NOT NULL,
    chat_id integer NOT NULL,
    role text NOT NULL,
    content text NOT NULL,
    sources text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: chat_messages_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.chat_messages_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: chat_messages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.chat_messages_id_seq OWNED BY public.chat_messages.id;


--
-- Name: chats; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chats (
    id integer NOT NULL,
    user_id integer NOT NULL,
    title text NOT NULL,
    status text DEFAULT 'active'::text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: chats_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.chats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: chats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.chats_id_seq OWNED BY public.chats.id;


--
-- Name: chunk_datasources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chunk_datasources (
    chunk_id text NOT NULL,
    datasource_id integer NOT NULL,
    embedding_model text DEFAULT ''::text NOT NULL
);


--
-- Name: cloud_audit_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cloud_audit_log (
    id integer NOT NULL,
    user_id integer,
    question_anonymized text NOT NULL,
    answer_anonymized text,
    provider text NOT NULL,
    model text NOT NULL,
    tokens_in integer DEFAULT 0,
    tokens_out integer DEFAULT 0,
    entities_anonymized integer DEFAULT 0,
    duration_ms integer DEFAULT 0,
    context_sources text,
    created_at timestamp with time zone DEFAULT now(),
    correlation_id text
);


--
-- Name: cloud_audit_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cloud_audit_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: cloud_audit_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cloud_audit_log_id_seq OWNED BY public.cloud_audit_log.id;


--
-- Name: config_changelog; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.config_changelog (
    id integer NOT NULL,
    user_id integer,
    username text DEFAULT ''::text NOT NULL,
    area text NOT NULL,
    action text NOT NULL,
    target_id text DEFAULT ''::text,
    target_label text DEFAULT ''::text,
    field_name text DEFAULT ''::text,
    old_value text,
    new_value text,
    created_at timestamp with time zone DEFAULT now(),
    source text DEFAULT 'gui'::text NOT NULL
);


--
-- Name: config_changelog_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.config_changelog_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: config_changelog_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.config_changelog_id_seq OWNED BY public.config_changelog.id;


--
-- Name: datasource_config; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.datasource_config (
    id integer NOT NULL,
    type text NOT NULL,
    webdav_url text DEFAULT ''::text,
    webdav_username text DEFAULT ''::text,
    webdav_password text DEFAULT ''::text,
    webdav_base_path text DEFAULT '/'::text,
    smb_server text DEFAULT ''::text,
    smb_share text DEFAULT ''::text,
    smb_username text DEFAULT ''::text,
    smb_password text DEFAULT ''::text,
    smb_domain text DEFAULT ''::text,
    smb_mount_point text DEFAULT '/mnt/kiara-smb'::text,
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT datasource_config_id_check CHECK ((id = 1))
);


--
-- Name: datasource_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.datasource_embeddings (
    id integer NOT NULL,
    datasource_id integer NOT NULL,
    embedding_model text NOT NULL,
    chunk_count integer DEFAULT 0,
    file_count integer DEFAULT 0,
    last_ingested_at text
);


--
-- Name: datasource_embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.datasource_embeddings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: datasource_embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.datasource_embeddings_id_seq OWNED BY public.datasource_embeddings.id;


--
-- Name: datasource_groups; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.datasource_groups (
    datasource_id integer NOT NULL,
    group_id integer NOT NULL
);


--
-- Name: datasource_users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.datasource_users (
    id integer NOT NULL,
    datasource_id integer NOT NULL,
    user_id integer NOT NULL
);


--
-- Name: datasource_users_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.datasource_users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: datasource_users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.datasource_users_id_seq OWNED BY public.datasource_users.id;


--
-- Name: datasources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.datasources (
    id integer NOT NULL,
    name text NOT NULL,
    type text DEFAULT 'local'::text NOT NULL,
    is_active boolean DEFAULT true,
    path text DEFAULT ''::text,
    webdav_url text DEFAULT ''::text,
    webdav_username text DEFAULT ''::text,
    webdav_password text DEFAULT ''::text,
    webdav_base_path text DEFAULT '/'::text,
    smb_server text DEFAULT ''::text,
    smb_share text DEFAULT ''::text,
    smb_username text DEFAULT ''::text,
    smb_password text DEFAULT ''::text,
    smb_domain text DEFAULT ''::text,
    smb_mount_point text DEFAULT ''::text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    ingestion_status text DEFAULT 'idle'::text,
    last_error text DEFAULT ''::text,
    processed_files integer DEFAULT 0,
    ds_chunk_size integer,
    ds_chunk_overlap integer,
    ds_file_types text,
    ds_exclude_patterns text,
    sync_interval text DEFAULT 'global'::text,
    next_sync_at timestamp with time zone,
    restriction_level text DEFAULT 'anonymize'::text NOT NULL,
    ds_chunk_strategy text DEFAULT 'auto'::text,
    ds_ddu_enabled boolean DEFAULT false,
    ds_vlm_enabled boolean DEFAULT false,
    imap_server text DEFAULT ''::text,
    imap_port integer DEFAULT 993,
    imap_ssl integer DEFAULT 1,
    imap_username text DEFAULT ''::text,
    imap_password text DEFAULT ''::text,
    imap_folders text DEFAULT 'INBOX'::text,
    imap_since_days integer DEFAULT 90,
    imap_download_attachments integer DEFAULT 1,
    imap_max_attachment_mb integer DEFAULT 25,
    sql_dsn text DEFAULT ''::text,
    sql_driver text DEFAULT 'auto'::text,
    sql_query text DEFAULT ''::text,
    sql_text_columns text DEFAULT ''::text,
    sql_id_column text DEFAULT 'id'::text,
    sql_timestamp_column text DEFAULT ''::text,
    sql_text_template text DEFAULT ''::text,
    connector_config text DEFAULT '{}'::text,
    progress_total integer DEFAULT 0,
    ds_index_attachments integer DEFAULT 1,
    ds_accept_all_text integer DEFAULT 0,
    ds_ddu_ocr_images integer DEFAULT 1,
    ds_contextual_retrieval integer DEFAULT 0,
    ds_enrichment_prompt_id integer,
    ds_enrichment_backend_id integer,
    ds_colbert_enabled boolean DEFAULT false,
    is_public boolean DEFAULT true
);


--
-- Name: datasources_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.datasources_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: datasources_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.datasources_id_seq OWNED BY public.datasources.id;


--
-- Name: file_hashes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.file_hashes (
    datasource_id integer NOT NULL,
    embedding_model text NOT NULL,
    file_path text NOT NULL,
    file_hash text NOT NULL,
    file_size bigint
);


--
-- Name: graph_edges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.graph_edges (
    id integer NOT NULL,
    source_id text NOT NULL,
    target_id text NOT NULL,
    relation text NOT NULL,
    properties text DEFAULT '{}'::text,
    chunk_id text,
    created_at timestamp with time zone DEFAULT now(),
    source text DEFAULT 'extraction'::text NOT NULL
);


--
-- Name: graph_edges_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.graph_edges_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: graph_edges_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.graph_edges_id_seq OWNED BY public.graph_edges.id;


--
-- Name: graph_nodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.graph_nodes (
    id text NOT NULL,
    entity_type text NOT NULL,
    label text NOT NULL,
    properties text DEFAULT '{}'::text,
    source text DEFAULT 'extraction'::text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: graph_node_chunks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.graph_node_chunks (
    node_id text NOT NULL,
    chunk_id text NOT NULL,
    CONSTRAINT graph_node_chunks_pkey PRIMARY KEY (node_id, chunk_id)
);


--
-- Name: graph_edge_chunks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.graph_edge_chunks (
    edge_id integer NOT NULL,
    chunk_id text NOT NULL,
    CONSTRAINT graph_edge_chunks_pkey PRIMARY KEY (edge_id, chunk_id)
);


--
-- Name: groups; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.groups (
    id integer NOT NULL,
    sid text,
    name text NOT NULL,
    cn text,
    dn text,
    member_count integer DEFAULT 0,
    imported_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: groups_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.groups_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: groups_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.groups_id_seq OWNED BY public.groups.id;


--
-- Name: hash_mode_meta; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.hash_mode_meta (
    datasource_id integer NOT NULL,
    embedding_model text NOT NULL,
    hash_mode text DEFAULT 'mtime'::text NOT NULL
);


--
-- Name: health_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.health_log (
    id integer NOT NULL,
    status text NOT NULL,
    components_json jsonb,
    degradation_mode text DEFAULT 'FULL'::text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: health_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.health_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: health_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.health_log_id_seq OWNED BY public.health_log.id;


--
-- Name: ingestion_file_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ingestion_file_logs (
    id integer NOT NULL,
    job_id integer NOT NULL,
    file_path text NOT NULL,
    file_name text NOT NULL,
    file_size bigint DEFAULT 0,
    file_ext text DEFAULT ''::text,
    parser text DEFAULT ''::text,
    parser_fallback integer DEFAULT 0,
    chunk_strategy text DEFAULT ''::text,
    chunk_count integer DEFAULT 0,
    status text DEFAULT 'success'::text NOT NULL,
    error_message text DEFAULT ''::text,
    error_phase text DEFAULT ''::text,
    duration_ms integer DEFAULT 0,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT ingestion_file_logs_status_check CHECK ((status = ANY (ARRAY['success'::text, 'error'::text, 'skipped'::text])))
);


--
-- Name: ingestion_file_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ingestion_file_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ingestion_file_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ingestion_file_logs_id_seq OWNED BY public.ingestion_file_logs.id;


--
-- Name: ingestion_jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ingestion_jobs (
    id integer NOT NULL,
    datasource_id integer NOT NULL,
    status text DEFAULT 'pending'::text NOT NULL,
    job_type text DEFAULT 'incremental'::text NOT NULL,
    triggered_by text DEFAULT 'manual'::text NOT NULL,
    embedding_model text DEFAULT ''::text,
    progress_processed integer DEFAULT 0,
    progress_total integer DEFAULT 0,
    queued_at timestamp with time zone DEFAULT now(),
    started_at timestamp with time zone DEFAULT now(),
    finished_at timestamp with time zone DEFAULT now(),
    stats_json jsonb DEFAULT '{}'::jsonb,
    error_message text DEFAULT ''::text,
    resumed_from_job integer,
    requested_action text,
    CONSTRAINT ingestion_jobs_job_type_check CHECK ((job_type = ANY (ARRAY['incremental'::text, 'full'::text]))),
    CONSTRAINT ingestion_jobs_triggered_by_check CHECK ((triggered_by = ANY (ARRAY['manual'::text, 'cron'::text, 'api'::text, 'auto-resume'::text, 'chat'::text])))
);


--
-- Name: ingestion_jobs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ingestion_jobs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ingestion_jobs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ingestion_jobs_id_seq OWNED BY public.ingestion_jobs.id;


--
-- Name: ki_backends; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ki_backends (
    id integer NOT NULL,
    category text NOT NULL,
    provider text NOT NULL,
    name text NOT NULL,
    url text DEFAULT ''::text,
    base_url text DEFAULT ''::text,
    model text NOT NULL,
    api_key text DEFAULT ''::text,
    is_active boolean DEFAULT true,
    priority integer DEFAULT 10,
    extra_config text DEFAULT '{}'::text,
    last_test_at timestamp with time zone,
    last_test_ok integer,
    last_test_error text DEFAULT ''::text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    trusted boolean DEFAULT false,
    CONSTRAINT ki_backends_category_check CHECK ((category = ANY (ARRAY['embedding'::text, 'llm'::text, 'cloud'::text, 'vlm'::text, 'ddu'::text]))),
    CONSTRAINT ki_backends_provider_check CHECK ((provider = ANY (ARRAY['ollama'::text, 'anthropic'::text, 'openai'::text, 'voyage'::text, 'docling'::text])))
);


--
-- Name: ki_backends_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ki_backends_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ki_backends_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ki_backends_id_seq OWNED BY public.ki_backends.id;


--
-- Name: ldap_config; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ldap_config (
    id integer NOT NULL,
    server_url text NOT NULL,
    port integer DEFAULT 389,
    use_ssl integer DEFAULT 0,
    bind_dn text NOT NULL,
    bind_password text NOT NULL,
    search_base text NOT NULL,
    group_dn text,
    group_name text,
    last_sync_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT ldap_config_id_check CHECK ((id = 1))
);


--
-- Name: mail_config; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mail_config (
    id integer NOT NULL,
    smtp_server text NOT NULL,
    smtp_port integer DEFAULT 587,
    encryption text DEFAULT 'starttls'::text,
    username text DEFAULT ''::text,
    password text DEFAULT ''::text,
    sender_address text NOT NULL,
    recipient_addresses text NOT NULL,
    notify_ollama_down integer DEFAULT 1,
    notify_datasource_down integer DEFAULT 1,
    notify_index_error integer DEFAULT 1,
    status_report_enabled boolean DEFAULT false,
    status_report_interval text DEFAULT 'weekly'::text,
    updated_at timestamp with time zone DEFAULT now(),
    notify_degradation integer DEFAULT 0,
    CONSTRAINT mail_config_id_check CHECK ((id = 1))
);


--
-- Name: onlyoffice_config; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.onlyoffice_config (
    id integer NOT NULL,
    document_server_url text NOT NULL,
    jwt_secret text NOT NULL,
    wopi_external_base_url text DEFAULT ''::text,
    proof_validation_enabled boolean DEFAULT true,
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT onlyoffice_config_id_check CHECK ((id = 1))
);


--
-- Name: prompt_defense_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_defense_events (
    id integer NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    user_id integer,
    source_type text NOT NULL,
    matched_patterns text DEFAULT '[]'::text NOT NULL,
    removed_snippets text DEFAULT '[]'::text NOT NULL,
    chars_removed integer DEFAULT 0 NOT NULL,
    original_length integer DEFAULT 0 NOT NULL,
    severity text DEFAULT 'niedrig'::text NOT NULL,
    question_hash text,
    chat_id integer,
    document_source text
);


--
-- Name: prompt_defense_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_defense_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompt_defense_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_defense_events_id_seq OWNED BY public.prompt_defense_events.id;


--
-- Name: prompt_template_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_template_history (
    id integer NOT NULL,
    template_id integer NOT NULL,
    content text NOT NULL,
    content_hash text NOT NULL,
    cloud_system_prompt text DEFAULT ''::text,
    changed_by text DEFAULT ''::text,
    change_reason text DEFAULT ''::text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: prompt_template_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_template_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompt_template_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_template_history_id_seq OWNED BY public.prompt_template_history.id;


--
-- Name: prompt_templates; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_templates (
    id integer NOT NULL,
    name text NOT NULL,
    label text DEFAULT ''::text,
    description text DEFAULT ''::text,
    content text NOT NULL,
    chain_type text DEFAULT 'standard'::text NOT NULL,
    is_cloud boolean DEFAULT false,
    show_in_chat integer DEFAULT 0,
    badge_color text DEFAULT ''::text,
    sort_order integer DEFAULT 0,
    is_system boolean DEFAULT false,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    final_prompt text DEFAULT ''::text,
    cloud_system_prompt text DEFAULT ''::text,
    CONSTRAINT prompt_templates_chain_type_check CHECK ((chain_type = ANY (ARRAY['standard'::text, 'unchained'::text, 'agent'::text, 'rewriter'::text, 'admin_agent'::text, 'enrichment'::text])))
);


--
-- Name: prompt_templates_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_templates_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompt_templates_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_templates_id_seq OWNED BY public.prompt_templates.id;


--
-- Name: query_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.query_log (
    id integer NOT NULL,
    user_id integer,
    question_hash text NOT NULL,
    question_excerpt text,
    answer_excerpt text,
    routed_to text DEFAULT 'local'::text NOT NULL,
    model_used text,
    sources_count integer DEFAULT 0,
    tools_used text,
    chunks_retrieved integer DEFAULT 0,
    entities_anonymized integer DEFAULT 0,
    duration_ms integer DEFAULT 0,
    chat_id integer,
    chain_type text,
    full_prompt text,
    full_answer text,
    tool_calls text,
    context_sources text,
    created_at timestamp with time zone DEFAULT now(),
    trace_id text,
    correlation_id text
);


--
-- Name: query_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.query_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: query_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.query_log_id_seq OWNED BY public.query_log.id;


--
-- Name: schema_version; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_version (
    version integer NOT NULL,
    applied_at text DEFAULT to_char((CURRENT_TIMESTAMP AT TIME ZONE 'localtime'::text), 'YYYY-MM-DD HH24:MI:SS'::text),
    description text
);


--
-- Name: sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sessions (
    id integer NOT NULL,
    user_id integer NOT NULL,
    token text NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    csrf_token_hash character(64)
);


--
-- Name: sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sessions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sessions_id_seq OWNED BY public.sessions.id;


--
-- Name: system_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.system_events (
    id bigint NOT NULL,
    ts timestamp with time zone DEFAULT now() NOT NULL,
    level text NOT NULL,
    source text DEFAULT 'security'::text NOT NULL,
    event_type text,
    reason text,
    user_name text,
    ip text,
    message text NOT NULL,
    details jsonb DEFAULT '{}'::jsonb NOT NULL
);


--
-- Name: system_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.system_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: system_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.system_events_id_seq OWNED BY public.system_events.id;


--
-- Name: system_metrics; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.system_metrics (
    id bigint NOT NULL,
    ts timestamp with time zone DEFAULT now() NOT NULL,
    resolution text DEFAULT 'raw'::text NOT NULL,
    cpu_percent real,
    ram_used_bytes bigint,
    ram_total_bytes bigint,
    swap_used_bytes bigint,
    swap_total_bytes bigint,
    load_1m real,
    load_5m real,
    load_15m real,
    net_sent_bps real,
    net_recv_bps real,
    disk_used_bytes bigint,
    disk_total_bytes bigint
);


--
-- Name: system_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.system_metrics_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: system_metrics_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.system_metrics_id_seq OWNED BY public.system_metrics.id;


--
-- Name: traces; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.traces (
    trace_id text NOT NULL,
    span_id text NOT NULL,
    parent_span_id text,
    name text NOT NULL,
    start_time_ns bigint NOT NULL,
    end_time_ns bigint NOT NULL,
    duration_ms double precision GENERATED ALWAYS AS ((((end_time_ns - start_time_ns))::numeric / 1000000.0)) STORED,
    attributes jsonb DEFAULT '{}'::jsonb NOT NULL,
    status text DEFAULT 'OK'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: user_groups; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_groups (
    user_id integer NOT NULL,
    group_id integer NOT NULL
);


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username text NOT NULL,
    password_hash text,
    vorname text DEFAULT ''::text,
    nachname text DEFAULT ''::text,
    email text DEFAULT ''::text,
    is_admin boolean DEFAULT false,
    is_active boolean DEFAULT true,
    auth_source text DEFAULT 'local'::text,
    must_change_password boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now(),
    ldap_dn text DEFAULT ''::text
);


--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: wopi_locks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.wopi_locks (
    file_id text NOT NULL,
    lock_id text NOT NULL,
    user_id integer NOT NULL,
    expires_at timestamp with time zone NOT NULL
);


--
-- Name: wopi_tokens; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.wopi_tokens (
    id integer NOT NULL,
    token text NOT NULL,
    file_id text NOT NULL,
    user_id integer NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    file_path text
);


--
-- Name: wopi_tokens_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.wopi_tokens_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: wopi_tokens_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.wopi_tokens_id_seq OWNED BY public.wopi_tokens.id;


--
-- Name: workflow_exports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.workflow_exports (
    id integer NOT NULL,
    user_id integer,
    filename text NOT NULL,
    format text NOT NULL,
    title text,
    size_bytes integer DEFAULT 0,
    template_used text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: workflow_exports_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.workflow_exports_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: workflow_exports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.workflow_exports_id_seq OWNED BY public.workflow_exports.id;


--
-- Name: admin_chat_confirmations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_confirmations ALTER COLUMN id SET DEFAULT nextval('public.admin_chat_confirmations_id_seq'::regclass);


--
-- Name: admin_chat_messages id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_messages ALTER COLUMN id SET DEFAULT nextval('public.admin_chat_messages_id_seq'::regclass);


--
-- Name: admin_chat_wizards id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_wizards ALTER COLUMN id SET DEFAULT nextval('public.admin_chat_wizards_id_seq'::regclass);


--
-- Name: admin_chats id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chats ALTER COLUMN id SET DEFAULT nextval('public.admin_chats_id_seq'::regclass);


--
-- Name: audit_anonymization id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_anonymization ALTER COLUMN id SET DEFAULT nextval('public.audit_anonymization_id_seq'::regclass);


--
-- Name: chat_messages id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages ALTER COLUMN id SET DEFAULT nextval('public.chat_messages_id_seq'::regclass);


--
-- Name: chats id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chats ALTER COLUMN id SET DEFAULT nextval('public.chats_id_seq'::regclass);


--
-- Name: cloud_audit_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cloud_audit_log ALTER COLUMN id SET DEFAULT nextval('public.cloud_audit_log_id_seq'::regclass);


--
-- Name: config_changelog id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_changelog ALTER COLUMN id SET DEFAULT nextval('public.config_changelog_id_seq'::regclass);


--
-- Name: datasource_embeddings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_embeddings ALTER COLUMN id SET DEFAULT nextval('public.datasource_embeddings_id_seq'::regclass);


--
-- Name: datasource_users id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_users ALTER COLUMN id SET DEFAULT nextval('public.datasource_users_id_seq'::regclass);


--
-- Name: datasources id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasources ALTER COLUMN id SET DEFAULT nextval('public.datasources_id_seq'::regclass);


--
-- Name: graph_edges id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_edges ALTER COLUMN id SET DEFAULT nextval('public.graph_edges_id_seq'::regclass);


--
-- Name: groups id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.groups ALTER COLUMN id SET DEFAULT nextval('public.groups_id_seq'::regclass);


--
-- Name: health_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.health_log ALTER COLUMN id SET DEFAULT nextval('public.health_log_id_seq'::regclass);


--
-- Name: ingestion_file_logs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_file_logs ALTER COLUMN id SET DEFAULT nextval('public.ingestion_file_logs_id_seq'::regclass);


--
-- Name: ingestion_jobs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_jobs ALTER COLUMN id SET DEFAULT nextval('public.ingestion_jobs_id_seq'::regclass);


--
-- Name: ki_backends id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ki_backends ALTER COLUMN id SET DEFAULT nextval('public.ki_backends_id_seq'::regclass);


--
-- Name: prompt_defense_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_defense_events ALTER COLUMN id SET DEFAULT nextval('public.prompt_defense_events_id_seq'::regclass);


--
-- Name: prompt_template_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_template_history ALTER COLUMN id SET DEFAULT nextval('public.prompt_template_history_id_seq'::regclass);


--
-- Name: prompt_templates id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_templates ALTER COLUMN id SET DEFAULT nextval('public.prompt_templates_id_seq'::regclass);


--
-- Name: query_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.query_log ALTER COLUMN id SET DEFAULT nextval('public.query_log_id_seq'::regclass);


--
-- Name: sessions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions ALTER COLUMN id SET DEFAULT nextval('public.sessions_id_seq'::regclass);


--
-- Name: system_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_events ALTER COLUMN id SET DEFAULT nextval('public.system_events_id_seq'::regclass);


--
-- Name: system_metrics id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_metrics ALTER COLUMN id SET DEFAULT nextval('public.system_metrics_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: wopi_tokens id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.wopi_tokens ALTER COLUMN id SET DEFAULT nextval('public.wopi_tokens_id_seq'::regclass);


--
-- Name: workflow_exports id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.workflow_exports ALTER COLUMN id SET DEFAULT nextval('public.workflow_exports_id_seq'::regclass);


--
-- Name: admin_chat_confirmations admin_chat_confirmations_action_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_confirmations
    ADD CONSTRAINT admin_chat_confirmations_action_id_key UNIQUE (action_id);


--
-- Name: admin_chat_confirmations admin_chat_confirmations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_confirmations
    ADD CONSTRAINT admin_chat_confirmations_pkey PRIMARY KEY (id);


--
-- Name: admin_chat_messages admin_chat_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_messages
    ADD CONSTRAINT admin_chat_messages_pkey PRIMARY KEY (id);


--
-- Name: admin_chat_wizards admin_chat_wizards_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_wizards
    ADD CONSTRAINT admin_chat_wizards_pkey PRIMARY KEY (id);


--
-- Name: admin_chats admin_chats_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chats
    ADD CONSTRAINT admin_chats_pkey PRIMARY KEY (id);


--
-- Name: ai_config ai_config_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ai_config
    ADD CONSTRAINT ai_config_pkey PRIMARY KEY (id);


--
-- Name: api_keys api_keys_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.api_keys
    ADD CONSTRAINT api_keys_pkey PRIMARY KEY (id);


--
-- Name: audit_anonymization audit_anonymization_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.audit_anonymization
    ADD CONSTRAINT audit_anonymization_pkey PRIMARY KEY (id);


--
-- Name: cache_datasources cache_datasources_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cache_datasources
    ADD CONSTRAINT cache_datasources_pkey PRIMARY KEY (datasource_id);


--
-- Name: cache_jobs cache_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cache_jobs
    ADD CONSTRAINT cache_jobs_pkey PRIMARY KEY (job_id);


--
-- Name: cache_vectordb cache_vectordb_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cache_vectordb
    ADD CONSTRAINT cache_vectordb_pkey PRIMARY KEY (id);


--
-- Name: chat_messages chat_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages
    ADD CONSTRAINT chat_messages_pkey PRIMARY KEY (id);


--
-- Name: chats chats_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chats
    ADD CONSTRAINT chats_pkey PRIMARY KEY (id);


--
-- Name: chunk_datasources chunk_datasources_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chunk_datasources
    ADD CONSTRAINT chunk_datasources_pkey PRIMARY KEY (chunk_id, datasource_id, embedding_model);


--
-- Name: cloud_audit_log cloud_audit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cloud_audit_log
    ADD CONSTRAINT cloud_audit_log_pkey PRIMARY KEY (id);


--
-- Name: config_changelog config_changelog_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.config_changelog
    ADD CONSTRAINT config_changelog_pkey PRIMARY KEY (id);


--
-- Name: datasource_config datasource_config_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_config
    ADD CONSTRAINT datasource_config_pkey PRIMARY KEY (id);


--
-- Name: datasource_embeddings datasource_embeddings_datasource_id_embedding_model_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_embeddings
    ADD CONSTRAINT datasource_embeddings_datasource_id_embedding_model_key UNIQUE (datasource_id, embedding_model);


--
-- Name: datasource_embeddings datasource_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_embeddings
    ADD CONSTRAINT datasource_embeddings_pkey PRIMARY KEY (id);


--
-- Name: datasource_groups datasource_groups_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_groups
    ADD CONSTRAINT datasource_groups_pkey PRIMARY KEY (datasource_id, group_id);


--
-- Name: datasource_users datasource_users_datasource_id_user_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_users
    ADD CONSTRAINT datasource_users_datasource_id_user_id_key UNIQUE (datasource_id, user_id);


--
-- Name: datasource_users datasource_users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_users
    ADD CONSTRAINT datasource_users_pkey PRIMARY KEY (id);


--
-- Name: datasources datasources_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasources
    ADD CONSTRAINT datasources_pkey PRIMARY KEY (id);


--
-- Name: file_hashes file_hashes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.file_hashes
    ADD CONSTRAINT file_hashes_pkey PRIMARY KEY (datasource_id, embedding_model, file_path);


--
-- Name: graph_edges graph_edges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_edges
    ADD CONSTRAINT graph_edges_pkey PRIMARY KEY (id);


--
-- Name: graph_nodes graph_nodes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_nodes
    ADD CONSTRAINT graph_nodes_pkey PRIMARY KEY (id);


--
-- Name: groups groups_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.groups
    ADD CONSTRAINT groups_pkey PRIMARY KEY (id);


--
-- Name: hash_mode_meta hash_mode_meta_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hash_mode_meta
    ADD CONSTRAINT hash_mode_meta_pkey PRIMARY KEY (datasource_id, embedding_model);


--
-- Name: health_log health_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.health_log
    ADD CONSTRAINT health_log_pkey PRIMARY KEY (id);


--
-- Name: ingestion_file_logs ingestion_file_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_file_logs
    ADD CONSTRAINT ingestion_file_logs_pkey PRIMARY KEY (id);


--
-- Name: ingestion_jobs ingestion_jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_jobs
    ADD CONSTRAINT ingestion_jobs_pkey PRIMARY KEY (id);


--
-- Name: ki_backends ki_backends_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ki_backends
    ADD CONSTRAINT ki_backends_pkey PRIMARY KEY (id);


--
-- Name: ldap_config ldap_config_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ldap_config
    ADD CONSTRAINT ldap_config_pkey PRIMARY KEY (id);


--
-- Name: mail_config mail_config_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mail_config
    ADD CONSTRAINT mail_config_pkey PRIMARY KEY (id);


--
-- Name: onlyoffice_config onlyoffice_config_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.onlyoffice_config
    ADD CONSTRAINT onlyoffice_config_pkey PRIMARY KEY (id);


--
-- Name: prompt_defense_events prompt_defense_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_defense_events
    ADD CONSTRAINT prompt_defense_events_pkey PRIMARY KEY (id);


--
-- Name: prompt_template_history prompt_template_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_template_history
    ADD CONSTRAINT prompt_template_history_pkey PRIMARY KEY (id);


--
-- Name: prompt_templates prompt_templates_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_templates
    ADD CONSTRAINT prompt_templates_name_key UNIQUE (name);


--
-- Name: prompt_templates prompt_templates_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_templates
    ADD CONSTRAINT prompt_templates_pkey PRIMARY KEY (id);


--
-- Name: query_log query_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.query_log
    ADD CONSTRAINT query_log_pkey PRIMARY KEY (id);


--
-- Name: schema_version schema_version_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_version
    ADD CONSTRAINT schema_version_pkey PRIMARY KEY (version);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_token_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_token_key UNIQUE (token);


--
-- Name: system_events system_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_events
    ADD CONSTRAINT system_events_pkey PRIMARY KEY (id);


--
-- Name: system_metrics system_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.system_metrics
    ADD CONSTRAINT system_metrics_pkey PRIMARY KEY (id);


--
-- Name: traces traces_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.traces
    ADD CONSTRAINT traces_pkey PRIMARY KEY (span_id);


--
-- Name: user_groups user_groups_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_groups
    ADD CONSTRAINT user_groups_pkey PRIMARY KEY (user_id, group_id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: wopi_locks wopi_locks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.wopi_locks
    ADD CONSTRAINT wopi_locks_pkey PRIMARY KEY (file_id);


--
-- Name: wopi_tokens wopi_tokens_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.wopi_tokens
    ADD CONSTRAINT wopi_tokens_pkey PRIMARY KEY (id);


--
-- Name: wopi_tokens wopi_tokens_token_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.wopi_tokens
    ADD CONSTRAINT wopi_tokens_token_key UNIQUE (token);


--
-- Name: workflow_exports workflow_exports_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.workflow_exports
    ADD CONSTRAINT workflow_exports_pkey PRIMARY KEY (id);


--
-- Name: idx_admin_chat_msgs_chat; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_admin_chat_msgs_chat ON public.admin_chat_messages USING btree (chat_id);


--
-- Name: idx_admin_chats_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_admin_chats_user ON public.admin_chats USING btree (user_id);


--
-- Name: idx_admin_confirm_chat; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_admin_confirm_chat ON public.admin_chat_confirmations USING btree (chat_id, status);


--
-- Name: idx_admin_confirm_expires; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_admin_confirm_expires ON public.admin_chat_confirmations USING btree (status, expires_at);


--
-- Name: idx_admin_wizard_active; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_admin_wizard_active ON public.admin_chat_wizards USING btree (chat_id) WHERE (status = 'collecting'::text);


--
-- Name: idx_admin_wizard_chat; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_admin_wizard_chat ON public.admin_chat_wizards USING btree (chat_id, status);


--
-- Name: idx_api_keys_hash; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_api_keys_hash ON public.api_keys USING btree (key_hash);


--
-- Name: idx_api_keys_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_api_keys_user ON public.api_keys USING btree (user_id);


--
-- Name: idx_cache_jobs_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cache_jobs_status ON public.cache_jobs USING btree (status);


--
-- Name: idx_ccl_area; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ccl_area ON public.config_changelog USING btree (area);


--
-- Name: idx_ccl_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ccl_created ON public.config_changelog USING btree (created_at);


--
-- Name: idx_cd_chunk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cd_chunk ON public.chunk_datasources USING btree (chunk_id);


--
-- Name: idx_cd_ds; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cd_ds ON public.chunk_datasources USING btree (datasource_id, embedding_model);


--
-- Name: idx_chat_messages_chat; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_chat_messages_chat ON public.chat_messages USING btree (chat_id);


--
-- Name: idx_chats_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_chats_user ON public.chats USING btree (user_id);


--
-- Name: idx_cloud_audit_corr_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_cloud_audit_corr_id ON public.cloud_audit_log USING btree (correlation_id) WHERE (correlation_id IS NOT NULL);


--
-- Name: idx_audit_anon_corr_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_audit_anon_corr_id ON public.audit_anonymization USING btree (correlation_id) WHERE (correlation_id IS NOT NULL);


--
-- Name: idx_cloud_audit_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cloud_audit_created ON public.cloud_audit_log USING btree (created_at);


--
-- Name: idx_cloud_audit_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cloud_audit_user ON public.cloud_audit_log USING btree (user_id);


--
-- Name: idx_datasource_groups_ds; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_datasource_groups_ds ON public.datasource_groups USING btree (datasource_id);


--
-- Name: idx_datasource_groups_group; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_datasource_groups_group ON public.datasource_groups USING btree (group_id);


--
-- Name: idx_groups_sid_unique; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_groups_sid_unique ON public.groups USING btree (sid) WHERE ((sid IS NOT NULL) AND (sid <> ''::text));


--
-- Name: idx_datasource_users_ds; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_datasource_users_ds ON public.datasource_users USING btree (datasource_id);


--
-- Name: idx_datasource_users_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_datasource_users_user ON public.datasource_users USING btree (user_id);


--
-- Name: idx_ds_embeddings_ds; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ds_embeddings_ds ON public.datasource_embeddings USING btree (datasource_id);


--
-- Name: idx_graph_edges_chunk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_graph_edges_chunk ON public.graph_edges USING btree (chunk_id);


--
-- Name: idx_graph_edges_relation; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_graph_edges_relation ON public.graph_edges USING btree (relation);


--
-- Name: idx_graph_edges_source; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_graph_edges_source ON public.graph_edges USING btree (source_id);


--
-- Name: idx_graph_edges_source_prov; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_graph_edges_source_prov ON public.graph_edges USING btree (source);


--
-- Name: idx_graph_edges_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_graph_edges_target ON public.graph_edges USING btree (target_id);


--
-- Name: idx_graph_edges_unique; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_graph_edges_unique ON public.graph_edges USING btree (source_id, target_id, relation, source);


--
-- Name: idx_graph_node_chunks_chunk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_graph_node_chunks_chunk ON public.graph_node_chunks USING btree (chunk_id);


--
-- Name: idx_graph_edge_chunks_chunk; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_graph_edge_chunks_chunk ON public.graph_edge_chunks USING btree (chunk_id);


--
-- Name: idx_graph_nodes_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_graph_nodes_type ON public.graph_nodes USING btree (entity_type);


--
-- Name: idx_health_log_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_health_log_created ON public.health_log USING btree (created_at);


--
-- Name: idx_ifl_job; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ifl_job ON public.ingestion_file_logs USING btree (job_id);


--
-- Name: idx_ifl_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ifl_status ON public.ingestion_file_logs USING btree (status);


--
-- Name: idx_ij_ds; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ij_ds ON public.ingestion_jobs USING btree (datasource_id);


--
-- Name: idx_ij_resumed_from; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ij_resumed_from ON public.ingestion_jobs USING btree (resumed_from_job);


--
-- Name: idx_ij_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ij_status ON public.ingestion_jobs USING btree (status);


--
-- Name: idx_pde_severity; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pde_severity ON public.prompt_defense_events USING btree (severity);


--
-- Name: idx_pde_source_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pde_source_type ON public.prompt_defense_events USING btree (source_type);


--
-- Name: idx_pde_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pde_timestamp ON public.prompt_defense_events USING btree ("timestamp");


--
-- Name: idx_pth_content_hash; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pth_content_hash ON public.prompt_template_history USING btree (content_hash);


--
-- Name: idx_pth_template_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pth_template_id ON public.prompt_template_history USING btree (template_id);


--
-- Name: idx_query_log_chat_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_chat_id ON public.query_log USING btree (chat_id);


--
-- Name: idx_query_log_corr_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_corr_id ON public.query_log USING btree (correlation_id) WHERE (correlation_id IS NOT NULL);


--
-- Name: idx_query_log_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_created ON public.query_log USING btree (created_at);


--
-- Name: idx_query_log_routed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_routed ON public.query_log USING btree (routed_to);


--
-- Name: idx_query_log_trace_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_trace_id ON public.query_log USING btree (trace_id);


--
-- Name: idx_query_log_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_user ON public.query_log USING btree (user_id);


--
-- Name: idx_sessions_token; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sessions_token ON public.sessions USING btree (token);


--
-- Name: idx_sysmetrics_res_ts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sysmetrics_res_ts ON public.system_metrics USING btree (resolution, ts);


--
-- Name: idx_system_events_level; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_system_events_level ON public.system_events USING btree (level);


--
-- Name: idx_system_events_reason; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_system_events_reason ON public.system_events USING btree (reason);


--
-- Name: idx_system_events_source; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_system_events_source ON public.system_events USING btree (source);


--
-- Name: idx_system_events_ts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_system_events_ts ON public.system_events USING btree (ts DESC);


--
-- Name: idx_system_events_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_system_events_type ON public.system_events USING btree (event_type);


--
-- Name: idx_system_events_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_system_events_user ON public.system_events USING btree (user_name);


--
-- Name: idx_traces_attributes; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_traces_attributes ON public.traces USING gin (attributes);


--
-- Name: idx_traces_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_traces_created ON public.traces USING btree (created_at DESC);


--
-- Name: idx_traces_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_traces_name ON public.traces USING btree (name);


--
-- Name: idx_traces_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_traces_status ON public.traces USING btree (status);


--
-- Name: idx_traces_trace_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_traces_trace_id ON public.traces USING btree (trace_id);


--
-- Name: idx_user_groups_group; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_user_groups_group ON public.user_groups USING btree (group_id);


--
-- Name: idx_user_groups_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_user_groups_user ON public.user_groups USING btree (user_id);


--
-- Name: idx_users_username_lower; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_users_username_lower ON public.users USING btree (lower(username));


--
-- Name: idx_wopi_tokens_token; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_wopi_tokens_token ON public.wopi_tokens USING btree (token);


--
-- Name: admin_chat_confirmations admin_chat_confirmations_chat_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_confirmations
    ADD CONSTRAINT admin_chat_confirmations_chat_id_fkey FOREIGN KEY (chat_id) REFERENCES public.admin_chats(id) ON DELETE CASCADE;


--
-- Name: admin_chat_confirmations admin_chat_confirmations_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_confirmations
    ADD CONSTRAINT admin_chat_confirmations_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: admin_chat_messages admin_chat_messages_chat_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_messages
    ADD CONSTRAINT admin_chat_messages_chat_id_fkey FOREIGN KEY (chat_id) REFERENCES public.admin_chats(id) ON DELETE CASCADE;


--
-- Name: admin_chat_wizards admin_chat_wizards_chat_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_wizards
    ADD CONSTRAINT admin_chat_wizards_chat_id_fkey FOREIGN KEY (chat_id) REFERENCES public.admin_chats(id) ON DELETE CASCADE;


--
-- Name: admin_chat_wizards admin_chat_wizards_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chat_wizards
    ADD CONSTRAINT admin_chat_wizards_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: admin_chats admin_chats_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.admin_chats
    ADD CONSTRAINT admin_chats_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: api_keys api_keys_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.api_keys
    ADD CONSTRAINT api_keys_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: chat_messages chat_messages_chat_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages
    ADD CONSTRAINT chat_messages_chat_id_fkey FOREIGN KEY (chat_id) REFERENCES public.chats(id) ON DELETE CASCADE;


--
-- Name: chats chats_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chats
    ADD CONSTRAINT chats_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: chunk_datasources chunk_datasources_datasource_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chunk_datasources
    ADD CONSTRAINT chunk_datasources_datasource_id_fkey FOREIGN KEY (datasource_id) REFERENCES public.datasources(id) ON DELETE CASCADE;


--
-- Name: datasource_embeddings datasource_embeddings_datasource_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_embeddings
    ADD CONSTRAINT datasource_embeddings_datasource_id_fkey FOREIGN KEY (datasource_id) REFERENCES public.datasources(id) ON DELETE CASCADE;


--
-- Name: datasource_groups datasource_groups_datasource_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_groups
    ADD CONSTRAINT datasource_groups_datasource_id_fkey FOREIGN KEY (datasource_id) REFERENCES public.datasources(id) ON DELETE CASCADE;


--
-- Name: datasource_groups datasource_groups_group_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_groups
    ADD CONSTRAINT datasource_groups_group_id_fkey FOREIGN KEY (group_id) REFERENCES public.groups(id) ON DELETE CASCADE;


--
-- Name: datasource_users datasource_users_datasource_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_users
    ADD CONSTRAINT datasource_users_datasource_id_fkey FOREIGN KEY (datasource_id) REFERENCES public.datasources(id) ON DELETE CASCADE;


--
-- Name: datasource_users datasource_users_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasource_users
    ADD CONSTRAINT datasource_users_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: datasources fk_ds_enrichment_backend; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasources
    ADD CONSTRAINT fk_ds_enrichment_backend FOREIGN KEY (ds_enrichment_backend_id) REFERENCES public.ki_backends(id) ON DELETE SET NULL;


--
-- Name: datasources fk_ds_enrichment_prompt; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasources
    ADD CONSTRAINT fk_ds_enrichment_prompt FOREIGN KEY (ds_enrichment_prompt_id) REFERENCES public.prompt_templates(id) ON DELETE SET NULL;


--
-- Name: graph_edges graph_edges_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_edges
    ADD CONSTRAINT graph_edges_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.graph_nodes(id) ON DELETE CASCADE;


--
-- Name: graph_edges graph_edges_target_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_edges
    ADD CONSTRAINT graph_edges_target_id_fkey FOREIGN KEY (target_id) REFERENCES public.graph_nodes(id) ON DELETE CASCADE;


--
-- Name: graph_node_chunks graph_node_chunks_node_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_node_chunks
    ADD CONSTRAINT graph_node_chunks_node_id_fkey FOREIGN KEY (node_id) REFERENCES public.graph_nodes(id) ON DELETE CASCADE;


--
-- Name: graph_edge_chunks graph_edge_chunks_edge_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_edge_chunks
    ADD CONSTRAINT graph_edge_chunks_edge_id_fkey FOREIGN KEY (edge_id) REFERENCES public.graph_edges(id) ON DELETE CASCADE;


--
-- Name: ingestion_jobs ingestion_jobs_datasource_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingestion_jobs
    ADD CONSTRAINT ingestion_jobs_datasource_id_fkey FOREIGN KEY (datasource_id) REFERENCES public.datasources(id) ON DELETE CASCADE;


--
-- Name: sessions sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: prompt_defense_events prompt_defense_events_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_defense_events
    ADD CONSTRAINT prompt_defense_events_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: prompt_template_history prompt_template_history_template_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_template_history
    ADD CONSTRAINT prompt_template_history_template_id_fkey FOREIGN KEY (template_id) REFERENCES public.prompt_templates(id) ON DELETE RESTRICT;


--
-- Name: user_groups user_groups_group_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_groups
    ADD CONSTRAINT user_groups_group_id_fkey FOREIGN KEY (group_id) REFERENCES public.groups(id) ON DELETE CASCADE;


--
-- Name: user_groups user_groups_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_groups
    ADD CONSTRAINT user_groups_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: wopi_locks wopi_locks_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.wopi_locks
    ADD CONSTRAINT wopi_locks_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: wopi_tokens wopi_tokens_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.wopi_tokens
    ADD CONSTRAINT wopi_tokens_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--
