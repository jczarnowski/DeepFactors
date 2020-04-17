#!/bin/bash
script_dir=$(dirname $0)
repo_dir=$(realpath ${script_dir}/..)

: ${python:=python} #python will take value "python" if not overriden
: ${build_dir:=build} # build_dir will take value "build" if not overriden
build_dir=$(realpath ${build_dir})
df_bin="${build_dir}/bin/df_demo"
scannet_script="${script_dir}/download-scannet.py"
reader_script="${script_dir}/sens_reader/reader.py"
download_dir="download"

# params
scene_id="scene0356_02"
export_depth=0

die() { echo -e "\n$*" 1>&2 ; exit 1; }

info() { echo -e "-- $*" 1>&2 ; }


function check_if_unpack_needed
{
  _scene_dir="$1"

  # when to unpack:
  # one of the dirs doesnt exist
  # export depth is requested and depth doesnt exist
  # if the dirst exist, the count doesnt match

  if [[ -d ${_scene_dir} ]]; then
    info "Scene output directory ${_scene_dir} already exists"

    # one of the dirs doesnt exist
    if [[ ! -d ${_scene_dir}/color 
       || ! -d ${_scene_dir}/pose 
       || ! -d ${_scene_dir}/intrinsic ]]; then
       return 1
    fi

    # all subdirs exist, but count doesn't match
    if [[ -d ${_scene_dir}/color 
       && -d ${_scene_dir}/pose 
       && -d ${_scene_dir}/intrinsic ]]; then

      # a heuristic to check if sens file has been unpacked
      num_color=$(ls -1q ${_scene_dir}/color | wc -l)
      num_poses=$(ls -1q ${_scene_dir}/pose | wc -l)
      if [[ $num_color -ne $num_poses ]]; then
        info "Number of frames and poses doesn't match"
        return 1
      fi
    fi

    # if export depth is requested and the directory doesn't exist
    if [[ ${export_depth} -eq 1 && ! -d ${_scene_dir}/depth ]]; then
      info "Export depth was requested and depth dir doesnt exist"
      return 1
    fi
  else
    return 1
  fi

  return 0
}

function download_scene_images 
{
  # Downloads the sens file for specified scene_id and
  # decompresses it into ${_out_dir}/${scene_id}
  _scene_id=$1
  _out_dir=$2

  sens_file="${_out_dir}/scans/${_scene_id}/${_scene_id}.sens"
  scene_out_dir="${_out_dir}/${_scene_id}"

  if [[ ! -f ${sens_file} ]]; then
    info "Downloading ${_scene_id}"
    yes | ${python} ${scannet_script} -o ${_out_dir} --id ${_scene_id} --type .sens || return 1
  fi

  # check if we need to unpack
  check_if_unpack_needed ${scene_out_dir}
  need_unpacking=$?

  if [[ ${need_unpacking} -eq 1 ]]; then
    # decompress .sens file
    info "Unpacking sens file ${sens_file}"
    reader_args=("--filename ${sens_file}" 
                 "--output_path ${scene_out_dir}"
                 "--export_intrinsics"
                 "--export_poses"
                 "--export_color_images")
    if [[ ${export_depth} -eq 1 ]]; then
      reader_args+=("--export_depth")
    fi
    ${python} ${reader_script} ${reader_args[@]} || return 1
  fi
}

function main() 
{
  # switch to top-level repo directory
  pushd ${repo_dir}

  info "Using default build directory $(realpath ${build_dir})"
  info "You can override it with the build_dir env variable"

  # check if the system was compiled
  err_msg="${df_bin} does not exist!\nMake sure you have compiled the project 
  and build_dir is correctly set" 
  [ ! -f "${df_bin}" ] && die ${err_msg}

  # check if download-scannet.py exists
  err_msg="${scannet_script} does not exist!\nMake sure you have have followed 
  the instructions in the readme"
  [ ! -f "${scannet_script}" ] && die ${err_msg}

  # patch download-scannet.py to work with python3
  patch -Nd ${script_dir} -r - < scripts/download_scannet_python3.patch

  # get a scannet sequence somewhere
  info "Getting ScanNet test sequence ${scene_id}. This might take a while"
  download_scene_images ${scene_id} ${download_dir} || die "Failed to download scene images!"

  # run the system on the sequence
  info "Starting DeepFactors"
  seq_dir="${download_dir}/${scene_id}"
  ${df_bin} --flagfile=data/flags/dataset_odom.flags --source_url=scannet://${seq_dir}

  popd
}

function usage()
{
  echo "Usage: $0 [--scene_id <scene_id>] [--extract_depth]"
}

function parse_args()
{
  options=$(getopt -o dv --long scene_id: --long export_depth --long help "$@")
  [ $? -eq 0 ] || {
    usage
    exit 1
  }
  eval set -- "$options"
  while true; do
    case "$1" in
      --scene_id)
        shift
        scene_id=$1
        ;;
      --export_depth)
        export_depth=1
        ;;
      --help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
    esac
    shift
  done
}

parse_args $0 "$@"

# run the entry point
main
