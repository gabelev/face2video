import get_frames_from_video
import turn_frames_into_video
import copy_sound_from_video
import automatic1111_api
import argparse
import os
import sys

class Face2Video:
    def __init__(self, input_face="", input_video="", processing_unit="CPU", input_type="Single Image"):
        self.input_face = input_face
        self.input_video = input_video
        self.processing_unit = processing_unit
        self.input_type = input_type

    def print_status(self, text):
        print(text)

    def split_video(self):
        if not self.input_video:
            self.print_status("No video selected")
            return False
        
        self.print_status("Splitting video into frames...")
        get_frames_from_video.get_frames(self.input_video)
        self.print_status("Finished splitting video into frames")
        self.print_status("Ready to swap face")
        return True

    def swap_face(self):
        if not self.input_video or not self.input_face:
            self.print_status("Select a face and a video")
            return False

        files = os.listdir("extracted_frames/")
        if not files:
            self.print_status("No frames found. Please split the video first.")
            return False

        if self.input_type == "Single Image":
            source_choice = 0
            input_model = ""
        else:
            source_choice = 1
            input_model = self.input_face
            self.input_face = ""

        for index, file in enumerate(files):
            file_path = os.path.join("extracted_frames/", file)
            automatic1111_api.api_change_face(file, self.input_face, input_model, file_path, self.processing_unit, source_choice)
            self.print_status(f"Finished image {index + 1} of {len(files)}")
        
        self.print_status("Finished swapping faces")
        self.print_status("Ready to merge frames")
        return True

    def delete_old_frames(self):
        paths = ["extracted_frames/", "finished_frames/"]
        for path in paths:
            if os.path.exists(path):
                files = os.listdir(path)
                for file in files:
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    else:
                        print(f"Skipped {file_path} as it's not a file.")
            else:
                print(f"The folder path '{path}' does not exist.")

    def merge_video(self):
        self.print_status("Creating video...")
        file_name_video = turn_frames_into_video.create_video("finished_frames/")
        self.print_status("Adding sound...")
        copy_sound_from_video.add_sound(self.input_video, file_name_video)
        self.print_status("Finished creating video!")

        self.delete_old_frames()
        return True

def process_video(args):
    face2video = Face2Video(
        input_face=args.face,
        input_video=args.video,
        processing_unit=args.processing_unit,
        input_type=args.input_type
    )

    if args.split:
        if not face2video.split_video():
            return False

    if args.swap:
        if not face2video.swap_face():
            return False

    if args.merge:
        if not face2video.merge_video():
            return False

    return True

def main():
    parser = argparse.ArgumentParser(description='Face2Video - Easy Faceswapping')
    parser.add_argument('--face', type=str, required=True, help='Path to the face image or model')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')
    parser.add_argument('--processing-unit', type=str, choices=['CPU', 'GPU (CUDA)'], default='CPU',
                      help='Processing unit to use (CPU or GPU)')
    parser.add_argument('--input-type', type=str, choices=['Single Image', 'Face Model'], default='Single Image',
                      help='Type of input face (Single Image or Face Model)')
    parser.add_argument('--split', action='store_true', help='Split video into frames')
    parser.add_argument('--swap', action='store_true', help='Swap face in frames')
    parser.add_argument('--merge', action='store_true', help='Merge frames into video')

    args = parser.parse_args()

    if not any([args.split, args.swap, args.merge]):
        parser.error("At least one of --split, --swap, or --merge must be specified")

    if not process_video(args):
        sys.exit(1)

if __name__ == "__main__":
    main()
