from src.osu.sections import Events


class TestEvents:
    def test_background(self):
        ev = Events(raw='0,0,"bg.jpg",0,0')
        assert ev.background is not None
        assert ev.background.filename == "bg.jpg"
        assert ev.background.x_offset == 0

    def test_background_with_offset(self):
        ev = Events(raw='0,0,"bg.jpg",10,-20')
        assert ev.background is not None
        assert ev.background.x_offset == 10
        assert ev.background.y_offset == -20

    def test_video(self):
        ev = Events(raw='Video,0,"intro.avi"')
        assert ev.video is not None
        assert ev.video.filename == "intro.avi"
        assert ev.video.start_time == 0

    def test_video_numeric_type(self):
        ev = Events(raw='1,500,"video.mp4"')
        assert ev.video is not None
        assert ev.video.filename == "video.mp4"
        assert ev.video.start_time == 500

    def test_breaks(self):
        ev = Events(raw="2,10000,15000")
        assert len(ev.breaks) == 1
        assert ev.breaks[0].start_time == 10000.0
        assert ev.breaks[0].end_time == 15000.0

    def test_multiple_breaks(self):
        ev = Events(raw="2,5000,8000\n2,20000,25000")
        assert len(ev.breaks) == 2

    def test_comments_skipped(self):
        ev = Events(raw='//Background and Video events\n0,0,"bg.jpg",0,0')
        assert ev.background is not None

    def test_empty(self):
        ev = Events()
        assert ev.background is None
        assert ev.video is None
        assert ev.breaks == []

    def test_background_roundtrip(self):
        ev = Events(raw='0,0,"bg.jpg",0,0')
        assert '"bg.jpg"' in str(ev)

    def test_break_roundtrip(self):
        ev = Events(raw="2,10000,15000")
        assert "2,10000,15000" in str(ev)
