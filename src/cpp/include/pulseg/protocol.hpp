/**
 * @file protocol.hpp
 * @brief Protocol parameters and the design-host bridge.
 *
 * C++ over pulseg_protocol.h and pulseg_bridge.h. Same calls, same order;
 * Bridge owns the child process, and the accessors throw pulseg::Error rather
 * than returning a code.
 */

#ifndef PULSEG_PROTOCOL_HPP
#define PULSEG_PROTOCOL_HPP

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulseg.h"

#include "error.hpp"
#include "types.hpp"

namespace pulseg
{
    /**
     * A protocol: the parameters a sequence takes, and their values.
     *
     * Fixed size and copyable, as the C struct is. Parameters are addressed by
     * id; `param_id()` maps a wire name to one.
     */
    class Protocol
    {
    public:
        Protocol()
        {
            proto_.count = 0;
        }

        /** Parameter id for a wire name, or -1 if the name is unknown. */
        static int param_id(const std::string& wire_name)
        {
            return pulseg_param_find(wire_name.c_str());
        }

        /** Wire name for a parameter id. Empty if the id is out of range. */
        static std::string wire_name(int param_id)
        {
            const char* name = pulseg_param_wire_name(param_id);
            return name ? std::string(name) : std::string();
        }

        /** Declared type of a parameter id: a PULSEG_PTYPE_* value. */
        static int param_type(int param_id)
        {
            return pulseg_param_get_type(param_id);
        }

        /** Number of populated parameters. */
        int size() const
        {
            return proto_.count;
        }

        /** The parameter ids this protocol carries, in order. */
        std::vector<int> keys() const
        {
            return std::vector<int>(proto_.keys, proto_.keys + proto_.count);
        }

        /** True if the protocol carries this parameter. */
        bool has(int param_id) const
        {
            return pulseg_protocol_find(&proto_, param_id) >= 0;
        }

        float get_float(int param_id) const
        {
            float value = 0.0f;
            check(pulseg_protocol_get_float(&proto_, &value, param_id));
            return value;
        }
        int get_int(int param_id) const
        {
            int value = 0;
            check(pulseg_protocol_get_int(&proto_, &value, param_id));
            return value;
        }
        bool get_bool(int param_id) const
        {
            int value = 0;
            check(pulseg_protocol_get_bool(&proto_, &value, param_id));
            return value != 0;
        }
        int get_config(int param_id) const
        {
            int value = 0;
            check(pulseg_protocol_get_config(&proto_, &value, param_id));
            return value;
        }
        int get_stringlist(int param_id) const
        {
            int index = 0;
            check(pulseg_protocol_get_stringlist(&proto_, &index, param_id));
            return index;
        }

        void set_float(int param_id, float value)
        {
            check(pulseg_protocol_set_float(&proto_, param_id, value));
        }
        void set_int(int param_id, int value)
        {
            check(pulseg_protocol_set_int(&proto_, param_id, value));
        }
        void set_bool(int param_id, bool value)
        {
            check(pulseg_protocol_set_bool(&proto_, param_id, value ? 1 : 0));
        }
        /** @param options Pipe-delimited option list, e.g. "off|low|high". */
        void set_stringlist(int param_id, int index, const std::string& options)
        {
            check(pulseg_protocol_set_stringlist(&proto_, param_id, index, options.c_str()));
        }

        /** Parse a protocol from a wire preamble. */
        static Protocol parse(const std::string& preamble)
        {
            Protocol protocol;
            check(pulseg_protocol_parse(&protocol.proto_, preamble.c_str()));
            return protocol;
        }

        /** Serialise to the wire form. */
        std::string serialize() const
        {
            std::vector<char> buffer(8192);
            const int written =
                pulseg_protocol_serialize(&proto_, buffer.data(), static_cast<int>(buffer.size()));
            check(written);
            return std::string(buffer.data());
        }

        pulseg_protocol* handle()
        {
            return &proto_;
        }
        const pulseg_protocol* handle() const
        {
            return &proto_;
        }

    private:
        pulseg_protocol proto_{};
    };

    /**
     * A process or pipe failure talking to the design host.
     *
     * Separate from pulseg::Error because what fails here is not a sequence
     * model: the C entry points report these as -1 with errno set rather than
     * as PULSEG_ERR_* codes, and this keeps that distinction.
     */
    class BridgeError : public std::runtime_error
    {
    public:
        explicit BridgeError(const std::string& what_failed)
            : std::runtime_error(what_failed + ": " + std::strerror(errno)), errno_(errno)
        {
        }

        int error_number() const noexcept
        {
            return errno_;
        }

    private:
        int errno_;
    };

    /** What VALIDATE answered. */
    struct ValidateResult
    {
        bool playable = false;
        float duration_s = 0.0f;
        std::string message;
    };

    /**
     * A live pypulseq_host child process.
     *
     * Sequence generation is Python; everything downstream of a .seq file is
     * this library. Bridge is the seam. It spawns the child on construction
     * and reaps it in the destructor.
     *
     * Failures here throw pulseg::BridgeError, not pulseg::Error: a broken
     * pipe is not a sequence-model failure, and the C entry points keep the
     * same distinction by returning -1 with errno set.
     */
    class Bridge
    {
    public:
        Bridge(const std::string& exe_path, const std::string& script_path, const Opts& opts)
        {
            opts_ = opts.to_c();
            if (pulseg_bridge_open_with_opts(
                    &bridge_,
                    exe_path.c_str(),
                    script_path.c_str(),
                    &opts_) != 0)
                throw BridgeError("could not start the design host");
            open_ = true;
        }

        ~Bridge()
        {
            close();
        }

        Bridge(Bridge&& o) noexcept : bridge_(o.bridge_), opts_(o.opts_), open_(o.open_)
        {
            o.open_ = false;
        }
        Bridge& operator=(Bridge&& o) noexcept
        {
            if (this != &o)
            {
                close();
                bridge_ = o.bridge_;
                opts_ = o.opts_;
                open_ = o.open_;
                o.open_ = false;
            }
            return *this;
        }
        Bridge(const Bridge&) = delete;
        Bridge& operator=(const Bridge&) = delete;

        /** Send QUIT, close the pipes and reap the child. Idempotent. */
        void close()
        {
            if (open_)
            {
                pulseg_bridge_close(&bridge_);
                open_ = false;
            }
        }

        /** LIST_PROTOCOL: the plugin's default protocol. */
        Protocol list_protocol()
        {
            Protocol protocol;
            if (pulseg_bridge_list_protocol(&bridge_, protocol.handle()) < 0)
                throw BridgeError("LIST_PROTOCOL");
            return protocol;
        }

        /** VALIDATE: whether a protocol is playable, and for how long. */
        ValidateResult validate(const Protocol& protocol)
        {
            std::vector<char> storage(512);
            pulseg_text_buffer message = PULSEG_TEXT_BUFFER_INIT;
            message.capacity = static_cast<int>(storage.size());
            message.data = storage.data();
            storage[0] = '\0';

            ValidateResult result;
            const int verdict =
                pulseg_bridge_validate(&bridge_, &result.duration_s, &message, protocol.handle());
            if (verdict < 0)
                throw BridgeError("VALIDATE");
            result.playable = (verdict == 1);
            result.message = storage.data();
            return result;
        }

        /** GENERATE: write a .seq for a protocol. */
        void generate(const Protocol& protocol, const std::string& output_path)
        {
            if (pulseg_bridge_generate(&bridge_, protocol.handle(), output_path.c_str()) != 0)
                throw BridgeError("GENERATE");
        }

        pulseg_bridge* handle()
        {
            return &bridge_;
        }

    private:
        pulseg_bridge bridge_{};
        pulseg_opts opts_{};
        bool open_ = false;
    };
} // namespace pulseg

#endif // PULSEG_PROTOCOL_HPP
