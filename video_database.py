import sqlite3
import uuid

class VideoEntry:
    def __init__(self, id, filename, time_started, animals=None, duration=None):
        self.id = id
        self.filename = filename
        self.time_started = time_started
        self.animals = animals
        self.duration = duration

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "time_started": self.time_started,
            "animals": self.animals,
            "duration": self.duration
        }

    def __repr__(self):
        return f"VideoEntry(video_id={self.video_id}, video_filename={self.video_filename}, time_started={self.time_started}, animals={self.animals}, duration={self.duration})"

class VideoDatabase:
    def __init__(self, db_name="videos.db"):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_table()

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.conn.row_factory = sqlite3.Row  # Enable access by column name
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise  # Re-raise the exception to prevent further execution

    def close(self):
        if self.conn:
            self.conn.close()

    def create_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    video_filename TEXT NOT NULL,
                    time_started INTEGER NOT NULL,
                    animals TEXT,
                    duration INTEGER
                )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")
            raise

    def insert_video(self, video_filename, time_started, animals=None, duration=None):
        video_id = str(uuid.uuid4())  # Generate a unique UUID
        try:
            self.cursor.execute("""
                INSERT INTO videos (video_id, video_filename, time_started, animals, duration)
                VALUES (?, ?, ?, ?, ?)
            """, (video_id, video_filename, time_started, str(animals) if animals else None, duration))
            self.conn.commit()
            return video_id  # Return the generated video_id
        except sqlite3.Error as e:
            print(f"Error inserting video: {e}")
            self.conn.rollback()  # Rollback in case of error
            return None

    def get_video(self, video_id) -> VideoEntry:
        try:
            self.cursor.execute("""
                SELECT video_id, video_filename, animals, duration, time_started FROM videos WHERE video_id = ?
            """, (video_id,))
            row = self.cursor.fetchone()
            if row:
                return VideoEntry(
                    id=row['video_id'],
                    filename=row['video_filename'],
                    time_started=row['time_started'],
                    animals=eval(row['animals']) if row['animals'] else None,
                    duration=row['duration']
                )
            else:
                return None
        except sqlite3.Error as e:
            print(f"Error getting video: {e}")
            return None

    def update_video(self, video_id, video_filename, animals, duration, time_started):
        try:
            self.cursor.execute("""
                UPDATE videos 
                SET video_filename = ?, animals = ?, duration = ?, time_started = ?
                WHERE video_id = ?
            """, (video_filename, str(animals), duration, time_started, video_id))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error updating video: {e}")
            self.conn.rollback()

    def update_video_animals(self, video_id, animals):
        try:
            self.cursor.execute("""
                UPDATE videos 
                SET animals = ?
                WHERE video_id = ?
            """, (str(animals), video_id))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error updating video animals: {e}")
            self.conn.rollback()

    def update_video_duration(self, video_id, duration):
        try:
            self.cursor.execute("""
                UPDATE videos 
                SET duration = ?
                WHERE video_id = ?
            """, (duration, video_id))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error updating video duration: {e}")
            self.conn.rollback()

    def delete_video(self, video_id):
        try:
            self.cursor.execute("""
                DELETE FROM videos WHERE video_id = ?
            """, (video_id,))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error deleting video: {e}")
            self.conn.rollback()

    def get_all_videos(self) -> list[VideoEntry]:
        try:
            self.cursor.execute("SELECT * FROM videos")
            rows = self.cursor.fetchall()
            videos = []
            for row in rows:
                videos.append(VideoEntry(
                    id=row['video_id'],
                    filename=row['video_filename'],
                    time_started=row['time_started'],
                    animals=eval(row['animals']) if row['animals'] else None,
                    duration=row['duration']
                ))
            return videos
        except sqlite3.Error as e:
            print(f"Error getting all videos: {e}")
            return []