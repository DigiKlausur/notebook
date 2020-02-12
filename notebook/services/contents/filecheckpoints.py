"""
File-based Checkpoints implementations.
"""
import os
import shutil
import time

from tornado.web import HTTPError

from .checkpoints import (
    Checkpoints,
    GenericCheckpointsMixin,
)
from .fileio import FileManagerMixin

from jupyter_core.utils import ensure_dir_exists
from ipython_genutils.py3compat import getcwd
from traitlets import Unicode, Int

from notebook import _tz as tz


class FileCheckpoints(FileManagerMixin, Checkpoints):
    """
    A Checkpoints that caches checkpoints for files in adjacent
    directories.

    Only works with FileContentsManager.  Use GenericFileCheckpoints if
    you want file-based checkpoints with another ContentsManager.
    """

    checkpoint_dir = Unicode(
        '.ipynb_checkpoints',
        config=True,
        help="""The directory name in which to keep file checkpoints

        This is a path relative to the file's own directory.

        By default, it is .ipynb_checkpoints
        """,
    )

    max_checkpoints = Int(
        5,
        config=True, 
        help="""Maximum number of checkpoints to save"""
    )

    min_seconds_between_checkpoints = Int(
        60, 
        config=True, 
        help="""If two checkpoints differ less than this, the existing one is overwritten"""
    )

    root_dir = Unicode(config=True)

    def _root_dir_default(self):
        try:
            return self.parent.root_dir
        except AttributeError:
            return getcwd()

    def get_checkpoint_ids(self):
        base_id = u'checkpoint'
        ids = []
        for i in range(self.max_checkpoints):
            ids.append('{}{}'.format(base_id, i))
        return ids

    def get_free_checkpoint_ids(self, path):
        path = path.strip('/')
        checkpoint_ids = self.get_checkpoint_ids()
        free_ids = []
        for checkpoint_id in checkpoint_ids:
            os_path = self.checkpoint_path(checkpoint_id, path)
            if not os.path.isfile(os_path):
                free_ids.append(checkpoint_id)
        return free_ids

    def timedelta(self, timestamp1, timestamp2):
        if timestamp1 > timestamp2:
            delta = timestamp1 - timestamp2
        else:
            delta = timestamp2 - timestamp1
        return delta.days*24*60*60 + delta.seconds

    # ContentsManager-dependent checkpoint API
    def create_checkpoint(self, contents_mgr, path):
        """Create a checkpoint."""
        checkpoint_id = u'checkpoint'
        # Check if there are free checkpoint ids
        free_ids = self.get_free_checkpoint_ids(path)
        if len(free_ids) > 0:
            checkpoint_id = free_ids[0]
        else:
            # Get the current checkpoints
            used_checkpoints = self.list_checkpoints(path)
            cur_time = tz.utcfromtimestamp(time.time())
            # If difference between last checkpoint and the current one is less than a delta
            # replace the last one, else replace the oldest one
            if self.timedelta(used_checkpoints[0]['last_modified'], cur_time) < self.min_seconds_between_checkpoints:
                checkpoint_id = used_checkpoints[0]['id']
            else:
                checkpoint_id = used_checkpoints[-1]['id']            
        
        src_path = contents_mgr._get_os_path(path)
        dest_path = self.checkpoint_path(checkpoint_id, path)
        self._copy(src_path, dest_path)
        return self.checkpoint_model(checkpoint_id, dest_path)

    def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """Restore a checkpoint."""
        src_path = self.checkpoint_path(checkpoint_id, path)
        dest_path = contents_mgr._get_os_path(path)
        self._copy(src_path, dest_path)

    # ContentsManager-independent checkpoint API
    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """Rename a checkpoint from old_path to new_path."""
        old_cp_path = self.checkpoint_path(checkpoint_id, old_path)
        new_cp_path = self.checkpoint_path(checkpoint_id, new_path)
        if os.path.isfile(old_cp_path):
            self.log.debug(
                "Renaming checkpoint %s -> %s",
                old_cp_path,
                new_cp_path,
            )
            with self.perm_to_403():
                shutil.move(old_cp_path, new_cp_path)

    def delete_checkpoint(self, checkpoint_id, path):
        """delete a file's checkpoint"""
        path = path.strip('/')
        cp_path = self.checkpoint_path(checkpoint_id, path)
        if not os.path.isfile(cp_path):
            self.no_such_checkpoint(path, checkpoint_id)

        self.log.debug("unlinking %s", cp_path)
        with self.perm_to_403():
            os.unlink(cp_path)

    def list_checkpoints(self, path):
        """list the checkpoints for a given file"""
        path = path.strip('/')
        checkpoint_ids = self.get_checkpoint_ids()
        checkpoints = []
        for checkpoint_id in checkpoint_ids:
            os_path = self.checkpoint_path(checkpoint_id, path)
            if os.path.isfile(os_path):
                checkpoints.append(self.checkpoint_model(checkpoint_id, os_path))
        return sorted(checkpoints, key=lambda x: x['last_modified'], reverse=True)

    # Checkpoint-related utilities
    def checkpoint_path(self, checkpoint_id, path):
        """find the path to a checkpoint"""
        path = path.strip('/')
        parent, name = ('/' + path).rsplit('/', 1)
        parent = parent.strip('/')
        basename, ext = os.path.splitext(name)
        filename = u"{name}-{checkpoint_id}{ext}".format(
            name=basename,
            checkpoint_id=checkpoint_id,
            ext=ext,
        )
        os_path = self._get_os_path(path=parent)
        cp_dir = os.path.join(os_path, self.checkpoint_dir)
        with self.perm_to_403():
            ensure_dir_exists(cp_dir)
        cp_path = os.path.join(cp_dir, filename)
        return cp_path

    def checkpoint_model(self, checkpoint_id, os_path):
        """construct the info dict for a given checkpoint"""
        stats = os.stat(os_path)
        last_modified = tz.utcfromtimestamp(stats.st_mtime)
        info = dict(
            id=checkpoint_id,
            last_modified=last_modified,
        )
        return info

    # Error Handling
    def no_such_checkpoint(self, path, checkpoint_id):
        raise HTTPError(
            404,
            u'Checkpoint does not exist: %s@%s' % (path, checkpoint_id)
        )


class GenericFileCheckpoints(GenericCheckpointsMixin, FileCheckpoints):
    """
    Local filesystem Checkpoints that works with any conforming
    ContentsManager.
    """
    def create_file_checkpoint(self, content, format, path):
        """Create a checkpoint from the current content of a file."""
        path = path.strip('/')
        # only the one checkpoint ID:
        checkpoint_id = u"checkpoint"
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        self.log.debug("creating checkpoint for %s", path)
        with self.perm_to_403():
            self._save_file(os_checkpoint_path, content, format=format)

        # return the checkpoint info
        return self.checkpoint_model(checkpoint_id, os_checkpoint_path)

    def create_notebook_checkpoint(self, nb, path):
        """Create a checkpoint from the current content of a notebook."""
        path = path.strip('/')
        # only the one checkpoint ID:
        checkpoint_id = u"checkpoint"
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)
        self.log.debug("creating checkpoint for %s", path)
        with self.perm_to_403():
            self._save_notebook(os_checkpoint_path, nb)

        # return the checkpoint info
        return self.checkpoint_model(checkpoint_id, os_checkpoint_path)

    def get_notebook_checkpoint(self, checkpoint_id, path):
        """Get a checkpoint for a notebook."""
        path = path.strip('/')
        self.log.info("restoring %s from checkpoint %s", path, checkpoint_id)
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)

        if not os.path.isfile(os_checkpoint_path):
            self.no_such_checkpoint(path, checkpoint_id)

        return {
            'type': 'notebook',
            'content': self._read_notebook(
                os_checkpoint_path,
                as_version=4,
            ),
        }

    def get_file_checkpoint(self, checkpoint_id, path):
        """Get a checkpoint for a file."""
        path = path.strip('/')
        self.log.info("restoring %s from checkpoint %s", path, checkpoint_id)
        os_checkpoint_path = self.checkpoint_path(checkpoint_id, path)

        if not os.path.isfile(os_checkpoint_path):
            self.no_such_checkpoint(path, checkpoint_id)

        content, format = self._read_file(os_checkpoint_path, format=None)
        return {
            'type': 'file',
            'content': content,
            'format': format,
        }
